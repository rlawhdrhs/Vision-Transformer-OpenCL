#pragma warning(disable : 4996)
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <CL/cl.h>
#include "ViT_opencl.h"
#include "Network.h"

#define img_size 224
#define patch_size 16
#define in_chans 3
#define num_classes 1000
#define embed_dim 768
#define num_heads 12
#define mlp_ratio 4.0
#define BATCH_SIZE_DEFAULT 16
#define MAX_PROFILE_KERNELS 32
#define TS 32

// 프로파일링 실행 1, 아니면 0
#define ENABLE_PROFILING 1

// --------------------------------------------------------
// 자료구조 및 전역 변수
// --------------------------------------------------------
typedef struct {
    cl_context context;
    cl_command_queue queues[2]; // Double Buffering을 위한 2개의 큐
    cl_program program;
    cl_kernel k_conv, k_flat, k_prep, k_ln, k_gemm, k_res, k_soft, k_gelu, k_lin, k_bias, k_score, k_mha_soft, k_context, k_cls_soft, k_fc1_tiled, k_fc2_tiled;
    cl_device_id device;
} OpenCL_Resources;

typedef struct {
    cl_mem batch_input;      // [Batch, 3, 224, 224]
    cl_mem layer0_out;       // Conv Out [Batch, 768, 14, 14]
    cl_mem layer1_out;       // Flatten [Batch, 196, 768]

    // Encoder
    cl_mem enc_in;           // [Batch, 197, 768]
    cl_mem enc_out;          // [Batch, 197, 768]
    cl_mem ln_buf;           // LayerNorm 결과 저장용
    cl_mem attn_res_buf;     // Attention 결과 저장용
    cl_mem mlp_res_buf;      // MLP 결과 저장용

    // MHA Internal
    cl_mem q_buf, k_buf, v_buf;
    cl_mem attn_score;       // [Batch, Heads, 197, 197]
    cl_mem attn_out_linear;  // MHA 최종 Linear 전 결과

    // MLP Internal
    cl_mem mlp_fc1_out;      // [Batch, 197, 3072]

    // Output Buffers (Pool에 포함시켜 관리)
    cl_mem logit_buf;        // [Batch, 197, 1000]
    cl_mem prob_buf;         // [Batch, 1000]
} ViT_Memory_Pool;

typedef struct {
    char name[64];        // 커널 이름
    double total_time_ms; // 누적 실행 시간
    long call_count;      // 호출 횟수
} KernelProfile;

KernelProfile g_profiler[MAX_PROFILE_KERNELS];
int g_profile_count = 0;

OpenCL_Resources g_opencl;
ViT_Memory_Pool g_mem_pools[2]; // 2개의 메모리 풀
cl_mem g_weight_buffers[152];   // 가중치는 Read-Only라 공유 가능

// --------------------------------------------------------
// Utility Functions
// --------------------------------------------------------
char *get_source_code(const char *file_name, size_t *len) {
    FILE *file = fopen(file_name, "rb");
    if (!file) { printf("Failed to open %s\n", file_name); exit(1); }
    fseek(file, 0, SEEK_END);
    *len = ftell(file);
    rewind(file);
    char *src = (char *)malloc(*len + 1);
    fread(src, *len, 1, file);
    src[*len] = '\0';
    fclose(file);
    return src;
}

void register_kernel_time(const char *name, cl_event event) {
#if ENABLE_PROFILING
    cl_ulong start, end;
    clWaitForEvents(1, &event);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);

    double ms = (double)(end - start) * 1e-6;

    int idx = -1;
    for (int i = 0; i < g_profile_count; i++) {
        if (strcmp(g_profiler[i].name, name) == 0) {
            idx = i;
            break;
        }
    }
    if (idx == -1) {
        if (g_profile_count < MAX_PROFILE_KERNELS) {
            idx = g_profile_count++;
            strcpy(g_profiler[idx].name, name);
            g_profiler[idx].total_time_ms = 0.0;
            g_profiler[idx].call_count = 0;
        }
        else {
            return;
        }
    }
    g_profiler[idx].total_time_ms += ms;
    g_profiler[idx].call_count++;
#endif
}

void print_profiling_stats() {
#if ENABLE_PROFILING
    printf("\n");
    printf("========================================================================\n");
    printf("                  OPENCL KERNEL PROFILING RESULTS                        \n");
    printf("========================================================================\n");
    printf(" %-20s | %8s | %15s | %12s \n", "Kernel Name", "Calls", "Total Time (ms)", "Avg Time (ms)");
    printf("------------------------------------------------------------------------\n");

    double grand_total = 0.0;
    for (int i = 0; i < g_profile_count; i++) {
        double avg = g_profiler[i].total_time_ms / g_profiler[i].call_count;
        printf(" %-20s | %8ld | %15.4f | %12.5f \n",
            g_profiler[i].name, g_profiler[i].call_count, g_profiler[i].total_time_ms, avg);
        grand_total += g_profiler[i].total_time_ms;
    }
    printf("------------------------------------------------------------------------\n");
    printf(" TOTAL GPU KERNEL TIME : %.4f sec\n", grand_total / 1000.0);
    printf("========================================================================\n\n");
#endif
}

void initialize_opencl() {
    cl_int err;
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &g_opencl.device, NULL);
    g_opencl.context = clCreateContext(NULL, 1, &g_opencl.device, NULL, NULL, &err);
    CHECK_ERROR(err);

    // 프로파일링 활성화 여부에 따른 Queue 속성
    cl_command_queue_properties props_val = 0;
#if ENABLE_PROFILING
    props_val = CL_QUEUE_PROFILING_ENABLE;
#endif
    cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, props_val, 0 };

    // Queue 2개 생성
    g_opencl.queues[0] = clCreateCommandQueueWithProperties(g_opencl.context, g_opencl.device, props, &err);
    CHECK_ERROR(err);
    g_opencl.queues[1] = clCreateCommandQueueWithProperties(g_opencl.context, g_opencl.device, props, &err);
    CHECK_ERROR(err);

    size_t len;
    char *source = get_source_code("kernel.cl", &len);
    g_opencl.program = clCreateProgramWithSource(g_opencl.context, 1, (const char **)&source, &len, &err);
    err = clBuildProgram(g_opencl.program, 1, &g_opencl.device, NULL, NULL, NULL);
    CHECK_ERROR(err);
    free(source);

    // 커널 생성
    g_opencl.k_conv = clCreateKernel(g_opencl.program, "Conv2d_Batched_Kernel", &err); CHECK_ERROR(err);
    g_opencl.k_flat = clCreateKernel(g_opencl.program, "FlattenTranspose_Batched_Kernel", &err); CHECK_ERROR(err);
    g_opencl.k_prep = clCreateKernel(g_opencl.program, "prepare_class_pos_kernel", &err); CHECK_ERROR(err);
    g_opencl.k_ln = clCreateKernel(g_opencl.program, "layer_norm_kernel", &err); CHECK_ERROR(err);
    g_opencl.k_gemm = clCreateKernel(g_opencl.program, "MHA_gemm_kernel", &err); CHECK_ERROR(err);
    g_opencl.k_res = clCreateKernel(g_opencl.program, "add_residual_kernel", &err); CHECK_ERROR(err);
    g_opencl.k_soft = clCreateKernel(g_opencl.program, "softmax_reduction_kernel", &err); CHECK_ERROR(err);
    g_opencl.k_gelu = clCreateKernel(g_opencl.program, "gelu_kernel", &err); CHECK_ERROR(err);
    g_opencl.k_lin = clCreateKernel(g_opencl.program, "linear_kernel_opt", &err); CHECK_ERROR(err);
    g_opencl.k_bias = clCreateKernel(g_opencl.program, "add_bias_broadcast_kernel", &err); CHECK_ERROR(err);
    g_opencl.k_score = clCreateKernel(g_opencl.program, "mha_score_kernel", &err); CHECK_ERROR(err);
    g_opencl.k_mha_soft = clCreateKernel(g_opencl.program, "mha_softmax_kernel", &err); CHECK_ERROR(err);
    g_opencl.k_context = clCreateKernel(g_opencl.program, "mha_context_kernel", &err); CHECK_ERROR(err);
    g_opencl.k_fc1_tiled = clCreateKernel(g_opencl.program, "fc1_kernel", &err); CHECK_ERROR(err);
    g_opencl.k_fc2_tiled = clCreateKernel(g_opencl.program, "fc2_kernel", &err); CHECK_ERROR(err);
    g_opencl.k_cls_soft = clCreateKernel(g_opencl.program, "extract_cls_softmax_kernel", &err); CHECK_ERROR(err);
}

// 인덱스를 받아 해당 풀을 초기화
void init_memory_pool_idx(int pool_idx, int batch_size) {
    cl_int err;
    int tokens = ((img_size / patch_size) * (img_size / patch_size)) + 1; // 197
    int hidden = (int)(embed_dim * mlp_ratio);

    size_t sz_in = sizeof(float) * batch_size * 3 * img_size * img_size;
    size_t sz_conv = sizeof(float) * batch_size * embed_dim * 14 * 14;
    size_t sz_flat = sizeof(float) * batch_size * 196 * embed_dim;
    size_t sz_emb = sizeof(float) * batch_size * tokens * embed_dim;
    size_t sz_score = sizeof(float) * batch_size * num_heads * tokens * tokens;
    size_t sz_mlp = sizeof(float) * batch_size * tokens * hidden;
    size_t sz_logit = sizeof(float) * batch_size * 197 * num_classes;
    size_t sz_prob = sizeof(float) * batch_size * num_classes;

    g_mem_pools[pool_idx].batch_input = clCreateBuffer(g_opencl.context, CL_MEM_READ_WRITE, sz_in, NULL, &err);
    g_mem_pools[pool_idx].layer0_out = clCreateBuffer(g_opencl.context, CL_MEM_READ_WRITE, sz_conv, NULL, &err);
    g_mem_pools[pool_idx].layer1_out = clCreateBuffer(g_opencl.context, CL_MEM_READ_WRITE, sz_flat, NULL, &err);

    g_mem_pools[pool_idx].enc_in = clCreateBuffer(g_opencl.context, CL_MEM_READ_WRITE, sz_emb, NULL, &err);
    g_mem_pools[pool_idx].enc_out = clCreateBuffer(g_opencl.context, CL_MEM_READ_WRITE, sz_emb, NULL, &err);
    g_mem_pools[pool_idx].ln_buf = clCreateBuffer(g_opencl.context, CL_MEM_READ_WRITE, sz_emb, NULL, &err);
    g_mem_pools[pool_idx].attn_res_buf = clCreateBuffer(g_opencl.context, CL_MEM_READ_WRITE, sz_emb, NULL, &err);
    g_mem_pools[pool_idx].mlp_res_buf = clCreateBuffer(g_opencl.context, CL_MEM_READ_WRITE, sz_emb, NULL, &err);

    g_mem_pools[pool_idx].q_buf = clCreateBuffer(g_opencl.context, CL_MEM_READ_WRITE, sz_emb, NULL, &err);
    g_mem_pools[pool_idx].k_buf = clCreateBuffer(g_opencl.context, CL_MEM_READ_WRITE, sz_emb, NULL, &err);
    g_mem_pools[pool_idx].v_buf = clCreateBuffer(g_opencl.context, CL_MEM_READ_WRITE, sz_emb, NULL, &err);
    g_mem_pools[pool_idx].attn_score = clCreateBuffer(g_opencl.context, CL_MEM_READ_WRITE, sz_score, NULL, &err);
    g_mem_pools[pool_idx].attn_out_linear = clCreateBuffer(g_opencl.context, CL_MEM_READ_WRITE, sz_emb, NULL, &err);

    g_mem_pools[pool_idx].mlp_fc1_out = clCreateBuffer(g_opencl.context, CL_MEM_READ_WRITE, sz_mlp, NULL, &err);
    g_mem_pools[pool_idx].logit_buf = clCreateBuffer(g_opencl.context, CL_MEM_READ_WRITE, sz_logit, NULL, &err);
    g_mem_pools[pool_idx].prob_buf = clCreateBuffer(g_opencl.context, CL_MEM_READ_WRITE, sz_prob, NULL, &err);
    CHECK_ERROR(err);
}

void load_all_weights(Network *networks) {
    cl_int err;
    for (int i = 0; i < 152; i++) {
        g_weight_buffers[i] = clCreateBuffer(g_opencl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            networks[i].size * sizeof(float), networks[i].data, &err);
        CHECK_ERROR(err);
    }
}

void release_resources() {
    // 1. 가중치 해제
    for (int i = 0; i < 152; i++) clReleaseMemObject(g_weight_buffers[i]);

    // 2. Memory Pools 해제 (2개 모두)
    for (int i = 0; i < 2; i++) {
        clReleaseMemObject(g_mem_pools[i].batch_input);
        clReleaseMemObject(g_mem_pools[i].layer0_out);
        clReleaseMemObject(g_mem_pools[i].layer1_out);
        clReleaseMemObject(g_mem_pools[i].enc_in);
        clReleaseMemObject(g_mem_pools[i].enc_out);
        clReleaseMemObject(g_mem_pools[i].ln_buf);
        clReleaseMemObject(g_mem_pools[i].attn_res_buf);
        clReleaseMemObject(g_mem_pools[i].mlp_res_buf);
        clReleaseMemObject(g_mem_pools[i].q_buf);
        clReleaseMemObject(g_mem_pools[i].k_buf);
        clReleaseMemObject(g_mem_pools[i].v_buf);
        clReleaseMemObject(g_mem_pools[i].attn_score);
        clReleaseMemObject(g_mem_pools[i].attn_out_linear);
        clReleaseMemObject(g_mem_pools[i].mlp_fc1_out);
        clReleaseMemObject(g_mem_pools[i].logit_buf);
        clReleaseMemObject(g_mem_pools[i].prob_buf);
    }

    // 3. 커널 및 프로그램 해제
    clReleaseKernel(g_opencl.k_conv); clReleaseKernel(g_opencl.k_flat);
    clReleaseKernel(g_opencl.k_prep); clReleaseKernel(g_opencl.k_ln);
    clReleaseKernel(g_opencl.k_gemm); clReleaseKernel(g_opencl.k_res);
    clReleaseKernel(g_opencl.k_soft); clReleaseKernel(g_opencl.k_gelu);
    clReleaseKernel(g_opencl.k_lin); clReleaseKernel(g_opencl.k_bias);
    clReleaseKernel(g_opencl.k_score); clReleaseKernel(g_opencl.k_mha_soft);
    clReleaseKernel(g_opencl.k_context); clReleaseKernel(g_opencl.k_cls_soft);
    clReleaseKernel(g_opencl.k_fc1_tiled); clReleaseKernel(g_opencl.k_fc2_tiled);
    clReleaseProgram(g_opencl.program);

    // 4. Queue 및 Context 해제
    clReleaseCommandQueue(g_opencl.queues[0]);
    clReleaseCommandQueue(g_opencl.queues[1]);
    clReleaseContext(g_opencl.context);
}

void run_conv2d(cl_command_queue queue, cl_mem in, cl_mem out, int w_idx, int b_idx, int batch_size) {
    int out_w = img_size / patch_size;
    int is = img_size, ps = patch_size, ic = in_chans, ed = embed_dim;
    size_t global[3] = { (size_t)embed_dim, (size_t)out_w, (size_t)(out_w * batch_size) };

    clSetKernelArg(g_opencl.k_conv, 0, sizeof(cl_mem), &in);
    clSetKernelArg(g_opencl.k_conv, 1, sizeof(cl_mem), &out);
    clSetKernelArg(g_opencl.k_conv, 2, sizeof(cl_mem), &g_weight_buffers[w_idx]);
    clSetKernelArg(g_opencl.k_conv, 3, sizeof(cl_mem), &g_weight_buffers[b_idx]);
    clSetKernelArg(g_opencl.k_conv, 4, sizeof(int), &is);
    clSetKernelArg(g_opencl.k_conv, 5, sizeof(int), &ps);
    clSetKernelArg(g_opencl.k_conv, 6, sizeof(int), &ic);
    clSetKernelArg(g_opencl.k_conv, 7, sizeof(int), &ed);
    clSetKernelArg(g_opencl.k_conv, 8, sizeof(int), &out_w);

#if ENABLE_PROFILING
    cl_event event;
    clEnqueueNDRangeKernel(queue, g_opencl.k_conv, 3, NULL, global, NULL, 0, NULL, &event);
    register_kernel_time("Conv2d", event);
    clReleaseEvent(event);
#else
    clEnqueueNDRangeKernel(queue, g_opencl.k_conv, 3, NULL, global, NULL, 0, NULL, NULL);
#endif
}

void run_flatten(cl_command_queue queue, cl_mem in, cl_mem out, int batch_size) {
    int out_w = img_size / patch_size;
    int num_patches = out_w * out_w;
    int ed = embed_dim;
    size_t global[2] = { (size_t)(num_patches * batch_size), (size_t)embed_dim };
    clSetKernelArg(g_opencl.k_flat, 0, sizeof(cl_mem), &in);
    clSetKernelArg(g_opencl.k_flat, 1, sizeof(cl_mem), &out);
    clSetKernelArg(g_opencl.k_flat, 2, sizeof(int), &out_w);
    clSetKernelArg(g_opencl.k_flat, 3, sizeof(int), &ed);
    clSetKernelArg(g_opencl.k_flat, 4, sizeof(int), &num_patches);

#if ENABLE_PROFILING
    cl_event event;
    clEnqueueNDRangeKernel(queue, g_opencl.k_flat, 2, NULL, global, NULL, 0, NULL, &event);
    register_kernel_time("flatten", event);
    clReleaseEvent(event);
#else
    clEnqueueNDRangeKernel(queue, g_opencl.k_flat, 2, NULL, global, NULL, 0, NULL, NULL);
#endif
}

void run_prepare_input(cl_command_queue queue, cl_mem flat_in, cl_mem enc_in, int cls_idx, int pos_idx, int batch_size) {
    int num_patches = (img_size / patch_size) * (img_size / patch_size);
    size_t global[3] = { (size_t)batch_size, (size_t)(num_patches + 1), (size_t)embed_dim };
    int ed = embed_dim;
    clSetKernelArg(g_opencl.k_prep, 0, sizeof(cl_mem), &flat_in);
    clSetKernelArg(g_opencl.k_prep, 1, sizeof(cl_mem), &enc_in);
    clSetKernelArg(g_opencl.k_prep, 2, sizeof(cl_mem), &g_weight_buffers[cls_idx]);
    clSetKernelArg(g_opencl.k_prep, 3, sizeof(cl_mem), &g_weight_buffers[pos_idx]);
    clSetKernelArg(g_opencl.k_prep, 4, sizeof(int), &batch_size);
    clSetKernelArg(g_opencl.k_prep, 5, sizeof(int), &ed);
    clSetKernelArg(g_opencl.k_prep, 6, sizeof(int), &num_patches);

#if ENABLE_PROFILING
    cl_event event;
    clEnqueueNDRangeKernel(queue, g_opencl.k_prep, 3, NULL, global, NULL, 0, NULL, &event);
    register_kernel_time("Token, Posemb", event);
    clReleaseEvent(event);
#else
    clEnqueueNDRangeKernel(queue, g_opencl.k_prep, 3, NULL, global, NULL, 0, NULL, NULL);
#endif
}

void run_layernorm(cl_command_queue queue, cl_mem in, cl_mem out, int w_idx, int b_idx, int total_tokens) {
    size_t local = 256;
    size_t global = total_tokens * local;
    size_t local_mem = (local * 2 + 2) * sizeof(float);
    int ed = embed_dim;
    clSetKernelArg(g_opencl.k_ln, 0, sizeof(cl_mem), &in);
    clSetKernelArg(g_opencl.k_ln, 1, sizeof(cl_mem), &out);
    clSetKernelArg(g_opencl.k_ln, 2, sizeof(cl_mem), &g_weight_buffers[w_idx]);
    clSetKernelArg(g_opencl.k_ln, 3, sizeof(cl_mem), &g_weight_buffers[b_idx]);
    clSetKernelArg(g_opencl.k_ln, 4, local_mem, NULL);
    clSetKernelArg(g_opencl.k_ln, 5, sizeof(int), &total_tokens);
    clSetKernelArg(g_opencl.k_ln, 6, sizeof(int), &ed);

#if ENABLE_PROFILING
    cl_event event;
    clEnqueueNDRangeKernel(queue, g_opencl.k_ln, 1, NULL, &global, &local, 0, NULL, &event);
    register_kernel_time("layer_norm", event);
    clReleaseEvent(event);
#else
    clEnqueueNDRangeKernel(queue, g_opencl.k_ln, 1, NULL, &global, &local, 0, NULL, NULL);
#endif
}

void run_linear(cl_command_queue queue, cl_mem in, cl_mem out, int w_idx, int b_idx, int tokens, int in_f, int out_f, int offset) {
    int WPT = 4;
    size_t global_col = ((out_f + TS - 1) / TS) * (TS / WPT);
    size_t global_row = ((tokens + TS - 1) / TS) * TS;

    size_t global[2] = { global_col, global_row };
    size_t local[2] = { TS / WPT, TS };

    clSetKernelArg(g_opencl.k_lin, 0, sizeof(int), &tokens);
    clSetKernelArg(g_opencl.k_lin, 1, sizeof(int), &out_f);
    clSetKernelArg(g_opencl.k_lin, 2, sizeof(int), &in_f);
    clSetKernelArg(g_opencl.k_lin, 3, sizeof(cl_mem), &in);
    clSetKernelArg(g_opencl.k_lin, 4, sizeof(cl_mem), &g_weight_buffers[w_idx]);
    clSetKernelArg(g_opencl.k_lin, 5, sizeof(cl_mem), &g_weight_buffers[b_idx]);
    clSetKernelArg(g_opencl.k_lin, 6, sizeof(cl_mem), &out);
    clSetKernelArg(g_opencl.k_lin, 7, sizeof(int), &offset);

#if ENABLE_PROFILING
    cl_event event;
    clEnqueueNDRangeKernel(queue, g_opencl.k_lin, 2, NULL, global, local, 0, NULL, &event);
    register_kernel_time("Linear", event);
    clReleaseEvent(event);
#else
    clEnqueueNDRangeKernel(queue, g_opencl.k_lin, 2, NULL, global, local, 0, NULL, NULL);
#endif
}

void run_residual(cl_command_queue queue, cl_mem in, cl_mem out, int total_elems) {
    size_t global = ((total_elems + 255) / 256) * 256;
    clSetKernelArg(g_opencl.k_res, 0, sizeof(cl_mem), &in);
    clSetKernelArg(g_opencl.k_res, 1, sizeof(cl_mem), &out);
    clSetKernelArg(g_opencl.k_res, 2, sizeof(int), &total_elems);

#if ENABLE_PROFILING
    cl_event event;
    clEnqueueNDRangeKernel(queue, g_opencl.k_res, 1, NULL, &global, NULL, 0, NULL, &event);
    register_kernel_time("residual", event);
    clReleaseEvent(event);
#else
    clEnqueueNDRangeKernel(queue, g_opencl.k_res, 1, NULL, &global, NULL, 0, NULL, NULL);
#endif
}

void run_gelu(cl_command_queue queue, cl_mem data, int total) {
    size_t global = ((total + 255) / 256) * 256;
    clSetKernelArg(g_opencl.k_gelu, 0, sizeof(cl_mem), &data);
    clSetKernelArg(g_opencl.k_gelu, 1, sizeof(int), &total);

#if ENABLE_PROFILING
    cl_event event;
    clEnqueueNDRangeKernel(queue, g_opencl.k_gelu, 1, NULL, &global, NULL, 0, NULL, &event);
    register_kernel_time("gelu", event);
    clReleaseEvent(event);
#else
    clEnqueueNDRangeKernel(queue, g_opencl.k_gelu, 1, NULL, &global, NULL, 0, NULL, NULL);
#endif
}


void MHA_batched(cl_command_queue queue, ViT_Memory_Pool *pool, cl_mem in, cl_mem out, int w_idx, int b_idx, int out_w_idx, int out_b_idx, int batch_size) {
    int tokens = ((img_size / patch_size) * (img_size / patch_size)) + 1;
    int total_tokens = tokens * batch_size;
    int head_dim = embed_dim / num_heads;
    float scale = 1.0f / sqrtf((float)head_dim);

    // Q, K, V 행렬 생성
    run_linear(queue, in, pool->q_buf, w_idx, b_idx, total_tokens, embed_dim, embed_dim, 0);
    run_linear(queue, in, pool->k_buf, w_idx, b_idx, total_tokens, embed_dim, embed_dim, embed_dim);
    run_linear(queue, in, pool->v_buf, w_idx, b_idx, total_tokens, embed_dim, embed_dim, 2 * embed_dim);

    size_t score_global[3] = {
        (size_t)((tokens + TS - 1) / TS * TS),
        (size_t)((tokens + TS - 1) / TS * TS),
        (size_t)(batch_size * num_heads)
    };
    size_t score_local[3] = { TS, TS, 1 };

    int nh = num_heads;

    // score = Q * K 행렬 곱 계산
    clSetKernelArg(g_opencl.k_score, 0, sizeof(cl_mem), &pool->q_buf);
    clSetKernelArg(g_opencl.k_score, 1, sizeof(cl_mem), &pool->k_buf);
    clSetKernelArg(g_opencl.k_score, 2, sizeof(cl_mem), &pool->attn_score);
    clSetKernelArg(g_opencl.k_score, 3, sizeof(int), &tokens);
    clSetKernelArg(g_opencl.k_score, 4, sizeof(int), &head_dim);
    clSetKernelArg(g_opencl.k_score, 5, sizeof(int), &nh);
    clSetKernelArg(g_opencl.k_score, 6, sizeof(float), &scale);

#if ENABLE_PROFILING
    cl_event e_score;
    clEnqueueNDRangeKernel(queue, g_opencl.k_score, 3, NULL, score_global, score_local, 0, NULL, &e_score);
    register_kernel_time("MHA Score", e_score); clReleaseEvent(e_score);
#else
    clEnqueueNDRangeKernel(queue, g_opencl.k_score, 3, NULL, score_global, score_local, 0, NULL, NULL);
#endif
    //softmax
    size_t soft_global[2] = { (size_t)(batch_size * num_heads), (size_t)tokens };
    clSetKernelArg(g_opencl.k_mha_soft, 0, sizeof(cl_mem), &pool->attn_score);
    clSetKernelArg(g_opencl.k_mha_soft, 1, sizeof(int), &tokens);

#if ENABLE_PROFILING
    cl_event e_soft;
    clEnqueueNDRangeKernel(queue, g_opencl.k_mha_soft, 2, NULL, soft_global, NULL, 0, NULL, &e_soft);
    register_kernel_time("MHA Softmax", e_soft); clReleaseEvent(e_soft);
#else
    clEnqueueNDRangeKernel(queue, g_opencl.k_mha_soft, 2, NULL, soft_global, NULL, 0, NULL, NULL);
#endif
    size_t ctx_global[3] = {
        (size_t)((head_dim + TS) / TS * TS),
        (size_t)((tokens + TS) / TS * TS),
        (size_t)(batch_size * num_heads)
    };
    size_t ctx_local[3] = { TS, TS, 1 };

    // attn * V 행렬 곱
    clSetKernelArg(g_opencl.k_context, 0, sizeof(cl_mem), &pool->attn_score);
    clSetKernelArg(g_opencl.k_context, 1, sizeof(cl_mem), &pool->v_buf);
    clSetKernelArg(g_opencl.k_context, 2, sizeof(cl_mem), &pool->attn_out_linear);
    clSetKernelArg(g_opencl.k_context, 3, sizeof(int), &tokens);
    clSetKernelArg(g_opencl.k_context, 4, sizeof(int), &head_dim);
    clSetKernelArg(g_opencl.k_context, 5, sizeof(int), &nh);

#if ENABLE_PROFILING
    cl_event e_ctx;
    clEnqueueNDRangeKernel(queue, g_opencl.k_context, 3, NULL, ctx_global, ctx_local, 0, NULL, &e_ctx);
    register_kernel_time("MHA Context", e_ctx); clReleaseEvent(e_ctx);
#else
    clEnqueueNDRangeKernel(queue, g_opencl.k_context, 3, NULL, ctx_global, ctx_local, 0, NULL, NULL);
#endif

    // 선형 변환
    run_linear(queue, pool->attn_out_linear, out, out_w_idx, out_b_idx, total_tokens, embed_dim, embed_dim, 0);
}

void MLP_batched(cl_command_queue queue, ViT_Memory_Pool *pool, cl_mem in, cl_mem out, int w1, int b1, int w2, int b2, int batch_size) {
    int tokens = ((img_size / patch_size) * (img_size / patch_size)) + 1;
    int total = tokens * batch_size;
    int hidden = (int)(embed_dim * mlp_ratio);

    run_linear(queue, in, pool->mlp_fc1_out, w1, b1, total, embed_dim, hidden, 0);
    run_gelu(queue, pool->mlp_fc1_out, total * hidden);
    run_linear(queue, pool->mlp_fc1_out, out, w2, b2, total, hidden, embed_dim, 0);
}


void MLP_batched2(cl_command_queue queue, ViT_Memory_Pool *pool, cl_mem in, cl_mem out, int w1_idx, int b1_idx, int w2_idx, int b2_idx, int batch_size)
{
    /* ----------- 여기서 필요한 변수들을 함수 맨 위에 전부 선언 ----------- */
    cl_mem d_fc1out;
    size_t l_fc[2];
    size_t g_fc1[2];
    int tokens_per_img;
    int total_tokens;
    int in_features;
    int hidden;
    int out_features;
    int total_hidden_elems;
    size_t g_gelu;
    size_t l_gelu;
    size_t g_fc2[2];
    cl_int err;


#ifdef ENABLE_PROFILING
    cl_event ev_fc1, ev_gelu, ev_fc2;
#endif

    /* ----------------------------- 계산 시작 ----------------------------- */

    tokens_per_img = ((img_size / patch_size) * (img_size / patch_size)) + 1;
    total_tokens = tokens_per_img * batch_size;

    in_features = embed_dim;
    hidden = (int)(embed_dim * mlp_ratio);
    out_features = embed_dim;

    /* 중간 결과 버퍼 */
    d_fc1out = pool->mlp_fc1_out;

    /* ----------------------------- FC1 (Tiled) ----------------------------- */
    l_fc[0] = 16;
    l_fc[1] = 16;

    g_fc1[0] = ((size_t)total_tokens + 16 - 1) / 16 * 16;
    g_fc1[1] = ((size_t)hidden + 16 - 1) / 16 * 16;

    clSetKernelArg(g_opencl.k_fc1_tiled, 0, sizeof(cl_mem), &in);
    clSetKernelArg(g_opencl.k_fc1_tiled, 1, sizeof(cl_mem), &g_weight_buffers[w1_idx]);
    clSetKernelArg(g_opencl.k_fc1_tiled, 2, sizeof(cl_mem), &g_weight_buffers[b1_idx]);
    clSetKernelArg(g_opencl.k_fc1_tiled, 3, sizeof(cl_mem), &d_fc1out);
    clSetKernelArg(g_opencl.k_fc1_tiled, 4, sizeof(int), &total_tokens);
    clSetKernelArg(g_opencl.k_fc1_tiled, 5, sizeof(int), &in_features);
    clSetKernelArg(g_opencl.k_fc1_tiled, 6, sizeof(int), &hidden);

#ifdef ENABLE_PROFILING
    err = clEnqueueNDRangeKernel(queue, g_opencl.k_fc1_tiled, 2, NULL, g_fc1, l_fc, 0, NULL, &ev_fc1);
    CHECK_ERROR(err);
    register_kernel_time("Linear", ev_fc1);
    clReleaseEvent(ev_fc1);
#else
    clEnqueueNDRangeKernel(g_opencl.queue, g_opencl.k_fc1_tiled, 2, NULL, g_fc1, l_fc, 0, NULL, NULL);
#endif

    /* ----------------------------- GELU ----------------------------- */
    total_hidden_elems = total_tokens * hidden;
    g_gelu = ((total_hidden_elems + 255) / 256) * 256;
    l_gelu = 256;

    clSetKernelArg(g_opencl.k_gelu, 0, sizeof(cl_mem), &d_fc1out);
    clSetKernelArg(g_opencl.k_gelu, 1, sizeof(int), &total_hidden_elems);

#ifdef ENABLE_PROFILING
    err = clEnqueueNDRangeKernel(queue, g_opencl.k_gelu, 1, NULL, &g_gelu, &l_gelu, 0, NULL, &ev_gelu);
    CHECK_ERROR(err);
    register_kernel_time("gelu", ev_gelu);
    clReleaseEvent(ev_gelu);
#else
    clEnqueueNDRangeKernel(g_opencl.queue, g_opencl.k_gelu_tiled, 1, NULL, &g_gelu, &l_gelu, 0, NULL, NULL);
#endif

    /* ----------------------------- FC2 ----------------------------- */
    g_fc2[0] = ((size_t)total_tokens + 16 - 1) / 16 * 16;
    g_fc2[1] = ((size_t)out_features + 16 - 1) / 16 * 16;

    clSetKernelArg(g_opencl.k_fc2_tiled, 0, sizeof(cl_mem), &d_fc1out);
    clSetKernelArg(g_opencl.k_fc2_tiled, 1, sizeof(cl_mem), &g_weight_buffers[w2_idx]);
    clSetKernelArg(g_opencl.k_fc2_tiled, 2, sizeof(cl_mem), &g_weight_buffers[b2_idx]);
    clSetKernelArg(g_opencl.k_fc2_tiled, 3, sizeof(cl_mem), &out);
    clSetKernelArg(g_opencl.k_fc2_tiled, 4, sizeof(int), &total_tokens);
    clSetKernelArg(g_opencl.k_fc2_tiled, 5, sizeof(int), &hidden);
    clSetKernelArg(g_opencl.k_fc2_tiled, 6, sizeof(int), &out_features);

#ifdef ENABLE_PROFILING
    err = clEnqueueNDRangeKernel(queue, g_opencl.k_fc2_tiled, 2, NULL, g_fc2, l_fc, 0, NULL, &ev_fc2);
    CHECK_ERROR(err);
    register_kernel_time("Linear", ev_fc2);
    clReleaseEvent(ev_fc2);
#else
    clEnqueueNDRangeKernel(g_opencl.queue, g_opencl.k_fc2_tiled, 2, NULL, g_fc2, l_fc, 0, NULL, NULL);
#endif
}

void Encoder_batched(cl_command_queue queue, ViT_Memory_Pool *pool, cl_mem in, cl_mem out, int *net_indices, int batch_size) {
    int tokens = ((img_size / patch_size) * (img_size / patch_size)) + 1;
    int total_elems = tokens * batch_size * embed_dim;

    // LN1 -> MHA -> Residual
    run_layernorm(queue, in, pool->ln_buf, net_indices[0], net_indices[1], tokens * batch_size);
    MHA_batched(queue, pool, pool->ln_buf, pool->attn_res_buf, net_indices[2], net_indices[3], net_indices[4], net_indices[5], batch_size);
    run_residual(queue, pool->attn_res_buf, in, total_elems);

    // LN2 -> MLP -> Residual
    run_layernorm(queue, in, pool->ln_buf, net_indices[6], net_indices[7], tokens * batch_size);
    MLP_batched2(queue, pool, pool->ln_buf, pool->mlp_res_buf, net_indices[8], net_indices[9], net_indices[10], net_indices[11], batch_size);

    // Copy for residual 2
    clEnqueueCopyBuffer(queue, in, out, 0, 0, total_elems * sizeof(float), 0, NULL, NULL);
    run_residual(queue, pool->mlp_res_buf, out, total_elems);
}


// --------------------------------------------------------
// Main Pipeline Function
// --------------------------------------------------------
void ViT_opencl(ImageData *image, Network *networks, float **probabilities) {
    int batch_size = BATCH_SIZE_DEFAULT;

    initialize_opencl();

    // 2개의 Pool 초기화
    init_memory_pool_idx(0, batch_size);
    init_memory_pool_idx(1, batch_size);

    load_all_weights(networks);

    int num_imgs = image[0].n;

    // Host Buffers (2 Sets for Ping-Pong)
    float *h_input[2];
    h_input[0] = (float *)malloc(sizeof(float) * batch_size * 3 * img_size * img_size);
    h_input[1] = (float *)malloc(sizeof(float) * batch_size * 3 * img_size * img_size);

    float *h_probs[2];
    h_probs[0] = (float *)malloc(sizeof(float) * batch_size * num_classes);
    h_probs[1] = (float *)malloc(sizeof(float) * batch_size * num_classes);

    int total_steps = (num_imgs + batch_size - 1) / batch_size;

    printf("Start ViT Pipeline (Total Batches: %d, Overlapping Enabled)\n", total_steps);

    // ------------------------------------------------------------
    // Pipelining Loop
    // ------------------------------------------------------------
    for (int step = 0; step < total_steps + 1; step++) {

        int curr_idx = step % 2;
        int prev_idx = (step - 1) % 2;

        int start_img_idx = step * batch_size;
        int current_batch = (start_img_idx + batch_size > num_imgs) ? (num_imgs - start_img_idx) : batch_size;


        if (step < total_steps) {
            for (int b = 0; b < current_batch; b++) {
                memcpy(h_input[curr_idx] + b * 3 * img_size * img_size,
                    image[start_img_idx + b].data,
                    sizeof(float) * 3 * img_size * img_size);
            }

            cl_command_queue Q = g_opencl.queues[curr_idx];
            ViT_Memory_Pool *P = &g_mem_pools[curr_idx];

            clEnqueueWriteBuffer(Q, P->batch_input, CL_FALSE, 0,
                current_batch * 3 * img_size * img_size * sizeof(float), h_input[curr_idx], 0, NULL, NULL);

            // Conv2d
            run_conv2d(Q, P->batch_input, P->layer0_out, 1, 2, current_batch);

            // Flatten
            run_flatten(Q, P->layer0_out, P->layer1_out, current_batch);

            // Pos Emb
            run_prepare_input(Q, P->layer1_out, P->enc_in, 0, 3, current_batch);

            // Encoder Loop
            cl_mem in_buf = P->enc_in;
            cl_mem out_buf = P->enc_out;
            for (int i = 0; i < 12; i++) {
                int base = 4 + i * 12;
                int indices[12];
                for (int k = 0; k < 12; k++) indices[k] = base + k;
                Encoder_batched(Q, P, in_buf, out_buf, indices, current_batch);
                cl_mem temp = in_buf; in_buf = out_buf; out_buf = temp;
            }

            // Head
            run_layernorm(Q, in_buf, P->enc_out, 148, 149, ((img_size / patch_size) * (img_size / patch_size) + 1) * current_batch);
            run_linear(Q, P->enc_out, P->logit_buf, 150, 151, ((img_size / patch_size) * (img_size / patch_size) + 1) * current_batch, embed_dim, num_classes, 0);

            // Softmax
            size_t soft_global = current_batch;
            int nc = num_classes;
            int seq_len = 197;
            clSetKernelArg(g_opencl.k_cls_soft, 0, sizeof(cl_mem), &P->logit_buf);
            clSetKernelArg(g_opencl.k_cls_soft, 1, sizeof(cl_mem), &P->prob_buf);
            clSetKernelArg(g_opencl.k_cls_soft, 2, sizeof(int), &nc);
            clSetKernelArg(g_opencl.k_cls_soft, 3, sizeof(int), &seq_len);

#if ENABLE_PROFILING
            cl_event e_soft;
            clEnqueueNDRangeKernel(Q, g_opencl.k_cls_soft, 1, NULL, &soft_global, NULL, 0, NULL, &e_soft);
            register_kernel_time("Cls Softmax", e_soft); clReleaseEvent(e_soft);
#else
            clEnqueueNDRangeKernel(Q, g_opencl.k_cls_soft, 1, NULL, &soft_global, NULL, 0, NULL, NULL);
#endif
            clEnqueueReadBuffer(Q, P->prob_buf, CL_FALSE, 0,
                sizeof(float) * current_batch * num_classes, h_probs[curr_idx], 0, NULL, NULL);
            clFlush(Q);
        }
        if (step > 0) {
            clFinish(g_opencl.queues[prev_idx]);

            int prev_step_idx = step - 1;
            int prev_start = prev_step_idx * batch_size;
            int prev_batch_cnt = (prev_start + batch_size > num_imgs) ? (num_imgs - prev_start) : batch_size;

            for (int b = 0; b < prev_batch_cnt; b++) {
                memcpy(probabilities[prev_start + b],
                    h_probs[prev_idx] + b * num_classes,
                    sizeof(float) * num_classes);
            }

            printf("Batch %d Finished (Imgs %d ~ %d)\n", prev_step_idx, prev_start, prev_start + prev_batch_cnt);
        }
    }
    free(h_input[0]); free(h_input[1]);
    free(h_probs[0]); free(h_probs[1]);

    print_profiling_stats();
    release_resources();
}

