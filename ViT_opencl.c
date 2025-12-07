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
#define ENABLE_PROFILING 1
// --------------------------------------------------------
// �ڷᱸ�� �� ���� ����
// --------------------------------------------------------
typedef struct {
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel k_conv, k_flat, k_prep, k_ln, k_gemm, k_res, k_soft, k_gelu, k_lin, k_bias, k_score, k_mha_soft, k_context, k_lin_reg, k_cls_soft;
    cl_device_id device;
} OpenCL_Resources;

typedef struct {
    cl_mem batch_input;      // [Batch, 3, 224, 224]
    cl_mem layer0_out;       // Conv Out [Batch, 768, 14, 14]
    cl_mem layer1_out;       // Flatten [Batch, 196, 768]

    // Encoder Ping-Pong Buffers
    cl_mem enc_in;           // [Batch, 197, 768]
    cl_mem enc_out;          // [Batch, 197, 768]
    cl_mem ln_buf;           // LayerNorm ��� �����
    cl_mem attn_res_buf;     // Attention ��� �����
    cl_mem mlp_res_buf;      // MLP ��� �����

    // MHA Internal
    cl_mem q_buf, k_buf, v_buf;
    cl_mem attn_score;       // [Batch, Heads, 197, 197]
    cl_mem attn_out_linear;  // MHA ���� Linear �� ���

    // MLP Internal
    cl_mem mlp_fc1_out;      // [Batch, 197, 3072]
} ViT_Memory_Pool;

typedef struct {
    char name[64];       // Ŀ�� �̸� (��: "Linear", "Softmax")
    double total_time_ms; // ���� ���� �ð� (�и���)
    long call_count;      // ȣ�� Ƚ��
} KernelProfile;

KernelProfile g_profiler[MAX_PROFILE_KERNELS];
int g_profile_count = 0;
OpenCL_Resources g_opencl;
ViT_Memory_Pool g_mem_pool;
cl_mem g_weight_buffers[152];


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

void initialize_opencl() {
    cl_int err;
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &g_opencl.device, NULL);
    g_opencl.context = clCreateContext(NULL, 1, &g_opencl.device, NULL, NULL, &err);
    CHECK_ERROR(err);
    cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    //g_opencl.queue = clCreateCommandQueueWithProperties(g_opencl.context, g_opencl.device, 0, &err);
    g_opencl.queue = clCreateCommandQueueWithProperties(g_opencl.context, g_opencl.device, props, &err);
    CHECK_ERROR(err);

    size_t len;
    char *source = get_source_code("kernel.cl", &len);
    g_opencl.program = clCreateProgramWithSource(g_opencl.context, 1, (const char **)&source, &len, &err);
    err = clBuildProgram(g_opencl.program, 1, &g_opencl.device, NULL, NULL, NULL);
    CHECK_ERROR(err);
    free(source);

    // Ŀ�� �̸� ����
    g_opencl.k_conv = clCreateKernel(g_opencl.program, "Conv2d_Batched_Kernel", &err);
    CHECK_ERROR(err);
    g_opencl.k_flat = clCreateKernel(g_opencl.program, "FlattenTranspose_Batched_Kernel", &err);
    CHECK_ERROR(err);
    g_opencl.k_prep = clCreateKernel(g_opencl.program, "prepare_class_pos_kernel", &err);
    CHECK_ERROR(err);
    g_opencl.k_ln = clCreateKernel(g_opencl.program, "layer_norm_kernel", &err);
    CHECK_ERROR(err);
    g_opencl.k_gemm = clCreateKernel(g_opencl.program, "MHA_gemm_kernel", &err);
    CHECK_ERROR(err);
    g_opencl.k_res = clCreateKernel(g_opencl.program, "add_residual_kernel", &err);
    CHECK_ERROR(err);
    g_opencl.k_soft = clCreateKernel(g_opencl.program, "softmax_reduction_kernel", &err);
    CHECK_ERROR(err);
    g_opencl.k_gelu = clCreateKernel(g_opencl.program, "gelu_kernel", &err);
    CHECK_ERROR(err);
    g_opencl.k_lin = clCreateKernel(g_opencl.program, "linear_tiled_float4", &err);
    CHECK_ERROR(err);
    g_opencl.k_bias = clCreateKernel(g_opencl.program, "add_bias_broadcast_kernel", &err); 
    CHECK_ERROR(err);

    //Tiling O 2d
    //g_opencl.k_score = clCreateKernel(g_opencl.program, "mha_score_kernel", &err);
    //Tiling x 3d
    g_opencl.k_score = clCreateKernel(g_opencl.program, "mha_score_kernel_3d", &err);
    CHECK_ERROR(err);

    g_opencl.k_mha_soft = clCreateKernel(g_opencl.program, "mha_softmax_kernel", &err);
    CHECK_ERROR(err);
    g_opencl.k_context = clCreateKernel(g_opencl.program, "mha_context_kernel", &err);
    CHECK_ERROR(err);
    g_opencl.k_cls_soft = clCreateKernel(g_opencl.program, "extract_cls_softmax_kernel", &err);
    CHECK_ERROR(err);
}

void register_kernel_time(const char *name, cl_event event) {
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
}

void print_profiling_stats() {
    printf("\n");
    printf("========================================================================\n");
    printf("                  OPENCL KERNEL PROFILING RESULTS                       \n");
    printf("========================================================================\n");
    printf(" %-20s | %8s | %15s | %12s \n", "Kernel Name", "Calls", "Total Time (ms)", "Avg Time (ms)");
    printf("------------------------------------------------------------------------\n");

    double grand_total = 0.0;
    for (int i = 0; i < g_profile_count; i++) {
        double avg = g_profiler[i].total_time_ms / g_profiler[i].call_count;
        printf(" %-20s | %8ld | %15.4f | %12.5f \n",
            g_profiler[i].name,
            g_profiler[i].call_count,
            g_profiler[i].total_time_ms,
            avg);
        grand_total += g_profiler[i].total_time_ms;
    }
    printf("------------------------------------------------------------------------\n");
    printf(" TOTAL GPU KERNEL TIME : %.4f sec\n", grand_total / 1000.0);
    printf("========================================================================\n\n");
}

void init_memory_pool(int batch_size) {
    cl_int err;
    int tokens = ((img_size / patch_size) * (img_size / patch_size)) + 1; // 197
    int hidden = (int)(embed_dim * mlp_ratio);

    size_t sz_in = sizeof(float) * batch_size * 3 * img_size * img_size;
    size_t sz_conv = sizeof(float) * batch_size * embed_dim * 14 * 14;
    size_t sz_flat = sizeof(float) * batch_size * 196 * embed_dim;
    size_t sz_emb = sizeof(float) * batch_size * tokens * embed_dim;
    size_t sz_score = sizeof(float) * batch_size * num_heads * tokens * tokens;
    size_t sz_mlp = sizeof(float) * batch_size * tokens * hidden;

    g_mem_pool.batch_input = clCreateBuffer(g_opencl.context, CL_MEM_READ_WRITE, sz_in, NULL, &err);
    g_mem_pool.layer0_out = clCreateBuffer(g_opencl.context, CL_MEM_READ_WRITE, sz_conv, NULL, &err);
    g_mem_pool.layer1_out = clCreateBuffer(g_opencl.context, CL_MEM_READ_WRITE, sz_flat, NULL, &err);

    g_mem_pool.enc_in = clCreateBuffer(g_opencl.context, CL_MEM_READ_WRITE, sz_emb, NULL, &err);
    g_mem_pool.enc_out = clCreateBuffer(g_opencl.context, CL_MEM_READ_WRITE, sz_emb, NULL, &err);
    g_mem_pool.ln_buf = clCreateBuffer(g_opencl.context, CL_MEM_READ_WRITE, sz_emb, NULL, &err);
    g_mem_pool.attn_res_buf = clCreateBuffer(g_opencl.context, CL_MEM_READ_WRITE, sz_emb, NULL, &err);
    g_mem_pool.mlp_res_buf = clCreateBuffer(g_opencl.context, CL_MEM_READ_WRITE, sz_emb, NULL, &err);

    g_mem_pool.q_buf = clCreateBuffer(g_opencl.context, CL_MEM_READ_WRITE, sz_emb, NULL, &err);
    g_mem_pool.k_buf = clCreateBuffer(g_opencl.context, CL_MEM_READ_WRITE, sz_emb, NULL, &err);
    g_mem_pool.v_buf = clCreateBuffer(g_opencl.context, CL_MEM_READ_WRITE, sz_emb, NULL, &err);
    g_mem_pool.attn_score = clCreateBuffer(g_opencl.context, CL_MEM_READ_WRITE, sz_score, NULL, &err);
    g_mem_pool.attn_out_linear = clCreateBuffer(g_opencl.context, CL_MEM_READ_WRITE, sz_emb, NULL, &err);

    g_mem_pool.mlp_fc1_out = clCreateBuffer(g_opencl.context, CL_MEM_READ_WRITE, sz_mlp, NULL, &err);
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
    for (int i = 0; i < 152; i++) clReleaseMemObject(g_weight_buffers[i]);
    clReleaseMemObject(g_mem_pool.batch_input);
    clReleaseMemObject(g_mem_pool.layer0_out);
    clReleaseMemObject(g_mem_pool.layer1_out);
    clReleaseMemObject(g_mem_pool.enc_in); clReleaseMemObject(g_mem_pool.enc_out);
    clReleaseMemObject(g_mem_pool.ln_buf); clReleaseMemObject(g_mem_pool.attn_res_buf);
    clReleaseMemObject(g_mem_pool.mlp_res_buf);
    clReleaseMemObject(g_mem_pool.q_buf); clReleaseMemObject(g_mem_pool.k_buf); clReleaseMemObject(g_mem_pool.v_buf);
    clReleaseMemObject(g_mem_pool.attn_score); clReleaseMemObject(g_mem_pool.attn_out_linear);
    clReleaseMemObject(g_mem_pool.mlp_fc1_out);

    clReleaseKernel(g_opencl.k_conv); clReleaseKernel(g_opencl.k_flat);
    clReleaseKernel(g_opencl.k_prep); clReleaseKernel(g_opencl.k_ln);
    clReleaseKernel(g_opencl.k_gemm); clReleaseKernel(g_opencl.k_res);
    clReleaseKernel(g_opencl.k_soft); clReleaseKernel(g_opencl.k_gelu);
    clReleaseKernel(g_opencl.k_lin); clReleaseKernel(g_opencl.k_bias);
    clReleaseProgram(g_opencl.program);
    clReleaseCommandQueue(g_opencl.queue);
    clReleaseContext(g_opencl.context);
}

// --------------------------------------------------------
// Batched Kernel Helpers
// --------------------------------------------------------
void run_conv2d(cl_mem in, cl_mem out, int w_idx, int b_idx, int batch_size) {
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
#ifdef ENABLE_PROFILING
    cl_event event;
    clEnqueueNDRangeKernel(g_opencl.queue, g_opencl.k_conv, 3, NULL, global, NULL, 0, NULL, &event);
    register_kernel_time("Conv2d", event);
    clReleaseEvent(event);
#else
    clEnqueueNDRangeKernel(g_opencl.queue, g_opencl.k_conv, 3, NULL, global, NULL, 0, NULL, NULL);
#endif
}

void run_flatten(cl_mem in, cl_mem out, int batch_size) {
    int out_w = img_size / patch_size;
    int num_patches = out_w * out_w;
    int ed = embed_dim;
    size_t global[2] = { (size_t)(num_patches * batch_size), (size_t)embed_dim };
    clSetKernelArg(g_opencl.k_flat, 0, sizeof(cl_mem), &in);
    clSetKernelArg(g_opencl.k_flat, 1, sizeof(cl_mem), &out);
    clSetKernelArg(g_opencl.k_flat, 2, sizeof(int), &out_w);
    clSetKernelArg(g_opencl.k_flat, 3, sizeof(int), &ed);
    clSetKernelArg(g_opencl.k_flat, 4, sizeof(int), &num_patches);
#ifdef ENABLE_PROFILING
    cl_event event;
    clEnqueueNDRangeKernel(g_opencl.queue, g_opencl.k_flat, 2, NULL, global, NULL, 0, NULL, &event);
    register_kernel_time("flatten", event);
    clReleaseEvent(event);
#else
    clEnqueueNDRangeKernel(g_opencl.queue, g_opencl.k_flat, 2, NULL, global, NULL, 0, NULL, NULL);
#endif
}

void run_prepare_input(cl_mem flat_in, cl_mem enc_in, int cls_idx, int pos_idx, int batch_size) {
    int num_patches = (img_size / patch_size) * (img_size / patch_size);
    size_t global[3] = { (size_t)batch_size, (size_t)(num_patches + 1), (size_t)embed_dim };
    int ed = embed_dim;
    clSetKernelArg(g_opencl.k_prep, 0, sizeof(cl_mem), &flat_in);
    clSetKernelArg(g_opencl.k_prep, 1, sizeof(cl_mem), &enc_in);
    clSetKernelArg(g_opencl.k_prep, 2, sizeof(cl_mem), &g_weight_buffers[cls_idx]); // Networks[0]
    clSetKernelArg(g_opencl.k_prep, 3, sizeof(cl_mem), &g_weight_buffers[pos_idx]); // Networks[3]
    clSetKernelArg(g_opencl.k_prep, 4, sizeof(int), &batch_size);
    clSetKernelArg(g_opencl.k_prep, 5, sizeof(int), &ed);
    clSetKernelArg(g_opencl.k_prep, 6, sizeof(int), &num_patches);

#ifdef ENABLE_PROFILING
    cl_event event;
    clEnqueueNDRangeKernel(g_opencl.queue, g_opencl.k_prep, 3, NULL, global, NULL, 0, NULL, &event);
    register_kernel_time("Token, Posemb", event);
    clReleaseEvent(event);
#else
    clEnqueueNDRangeKernel(g_opencl.queue, g_opencl.k_prep, 3, NULL, global, NULL, 0, NULL, NULL);
#endif
}

void run_layernorm(cl_mem in, cl_mem out, int w_idx, int b_idx, int total_tokens) {
    size_t local = 256;
    size_t global = total_tokens * local;
    size_t local_mem = (local * 2 + 2) * sizeof(float);
    int offset = 0;
    int ed = embed_dim;
    clSetKernelArg(g_opencl.k_ln, 0, sizeof(cl_mem), &in);
    clSetKernelArg(g_opencl.k_ln, 1, sizeof(cl_mem), &out);
    clSetKernelArg(g_opencl.k_ln, 2, sizeof(cl_mem), &g_weight_buffers[w_idx]);
    clSetKernelArg(g_opencl.k_ln, 3, sizeof(cl_mem), &g_weight_buffers[b_idx]);
    clSetKernelArg(g_opencl.k_ln, 4, local_mem, NULL);
    clSetKernelArg(g_opencl.k_ln, 5, sizeof(int), &total_tokens);
    clSetKernelArg(g_opencl.k_ln, 6, sizeof(int), &ed);
#ifdef ENABLE_PROFILING
    cl_event event;
    clEnqueueNDRangeKernel(g_opencl.queue, g_opencl.k_ln, 1, NULL, &global, &local, 0, NULL, &event);
    register_kernel_time("layer_norm", event);
    clReleaseEvent(event);
#else
    clEnqueueNDRangeKernel(g_opencl.queue, g_opencl.k_ln, 1, NULL, &global, &local, 0, NULL, NULL);
#endif
}

//void run_linear(cl_mem in, cl_mem out, int w_idx, int b_idx, int tokens, int in_f, int out_f, int offset) { 
//    int TS = 16;
//    //cl_event event;
//    size_t global[2] = {
//        (size_t)(((out_f + TS - 1) / TS) * TS),
//        (size_t)(((tokens + TS - 1) / TS) * TS)
//    };
//    size_t local[2] = { (size_t)TS, (size_t)TS };
//
//    clSetKernelArg(g_opencl.k_lin, 0, sizeof(int), &tokens); // M
//    clSetKernelArg(g_opencl.k_lin, 1, sizeof(int), &out_f);  // N
//    clSetKernelArg(g_opencl.k_lin, 2, sizeof(int), &in_f);   // K
//    clSetKernelArg(g_opencl.k_lin, 3, sizeof(cl_mem), &in);
//    clSetKernelArg(g_opencl.k_lin, 4, sizeof(cl_mem), &g_weight_buffers[w_idx]);
//    clSetKernelArg(g_opencl.k_lin, 5, sizeof(cl_mem), &g_weight_buffers[b_idx]);
//    clSetKernelArg(g_opencl.k_lin, 6, sizeof(cl_mem), &out);
//    clSetKernelArg(g_opencl.k_lin, 7, sizeof(int), &offset);
//#ifdef ENABLE_PROFILING
//    cl_event event;
//    clEnqueueNDRangeKernel(g_opencl.queue, g_opencl.k_lin, 2, NULL, global, local, 0, NULL, &event);
//    register_kernel_time("linear", event);
//    clReleaseEvent(event);
//#else
//    clEnqueueNDRangeKernel(g_opencl.queue, g_opencl.k_lin, 2, NULL, global, local, 0, NULL, NULL);
//#endif
//}

//void run_linear(cl_mem in, cl_mem out, int w_idx, int b_idx, int tokens, int in_f, int out_f, int offset, int use_gelu) {
//    int TS = 32;
//    int WPT = 4;
//
//    size_t global_M = ((tokens + TS - 1) / TS) * TS;
//    size_t global_N = ((out_f + TS - 1) / TS) * (TS / WPT);
//
//    size_t global[2] = { global_N, global_M };
//    size_t local[2] = { (size_t)(TS / WPT), (size_t)TS };
//
//    clSetKernelArg(g_opencl.k_lin_reg, 0, sizeof(int), &tokens);
//    clSetKernelArg(g_opencl.k_lin_reg, 1, sizeof(int), &out_f);
//    clSetKernelArg(g_opencl.k_lin_reg, 2, sizeof(int), &in_f);
//    clSetKernelArg(g_opencl.k_lin_reg, 3, sizeof(cl_mem), &in);
//    clSetKernelArg(g_opencl.k_lin_reg, 4, sizeof(cl_mem), &g_weight_buffers[w_idx]);
//    clSetKernelArg(g_opencl.k_lin_reg, 5, sizeof(cl_mem), &g_weight_buffers[b_idx]);
//    clSetKernelArg(g_opencl.k_lin_reg, 6, sizeof(cl_mem), &out);
//    clSetKernelArg(g_opencl.k_lin_reg, 7, sizeof(int), &offset);
//    clSetKernelArg(g_opencl.k_lin_reg, 8, sizeof(int), &use_gelu);
//
//#ifdef ENABLE_PROFILING
//    cl_event event;
//    clEnqueueNDRangeKernel(g_opencl.queue, g_opencl.k_lin_reg, 2, NULL, global, local, 0, NULL, &event);
//    register_kernel_time(use_gelu ? "Linear+GELU" : "Linear", event);
//    clReleaseEvent(event);
//#else
//    clEnqueueNDRangeKernel(g_opencl.queue, g_opencl.k_lin_reg, 2, NULL, global, local, 0, NULL, NULL);
//#endif
//}

void run_linear(cl_mem in, cl_mem out, int w_idx, int b_idx, int tokens, int in_f, int out_f, int offset) {
    int TS = 32;
    int WPT = 4;
    size_t global_col = ((out_f + TS - 1) / TS) * (TS / WPT);
    size_t global_row = ((tokens + TS - 1) / TS) * TS;

    size_t global[2] = { global_col, global_row };
    size_t local[2] = { 8, 32 };

    clSetKernelArg(g_opencl.k_lin, 0, sizeof(int), &tokens);
    clSetKernelArg(g_opencl.k_lin, 1, sizeof(int), &out_f);
    clSetKernelArg(g_opencl.k_lin, 2, sizeof(int), &in_f);
    clSetKernelArg(g_opencl.k_lin, 3, sizeof(cl_mem), &in);
    clSetKernelArg(g_opencl.k_lin, 4, sizeof(cl_mem), &g_weight_buffers[w_idx]);
    clSetKernelArg(g_opencl.k_lin, 5, sizeof(cl_mem), &g_weight_buffers[b_idx]);
    clSetKernelArg(g_opencl.k_lin, 6, sizeof(cl_mem), &out);
    clSetKernelArg(g_opencl.k_lin, 7, sizeof(int), &offset);
#ifdef ENABLE_PROFILING
    cl_event event;
    cl_int err = clEnqueueNDRangeKernel(g_opencl.queue, g_opencl.k_lin, 2, NULL, global, local, 0, NULL, &event);
    register_kernel_time("Linear", event);
    clReleaseEvent(event);
#else
    clEnqueueNDRangeKernel(g_opencl.queue, g_opencl.k_lin, 2, NULL, global, local, 0, NULL, NULL);
#endif
}

void run_add_bias(cl_mem data, cl_mem bias_buf, int rows, int cols, int bias_offset) {
    size_t global[2] = { (size_t)rows, (size_t)cols };
    clSetKernelArg(g_opencl.k_bias, 0, sizeof(cl_mem), &data);
    clSetKernelArg(g_opencl.k_bias, 1, sizeof(cl_mem), &bias_buf);
    clSetKernelArg(g_opencl.k_bias, 2, sizeof(int), &rows);
    clSetKernelArg(g_opencl.k_bias, 3, sizeof(int), &cols);
    clSetKernelArg(g_opencl.k_bias, 4, sizeof(int), &bias_offset);

#ifdef ENABLE_PROFILING
    cl_event event;
    clEnqueueNDRangeKernel(g_opencl.queue, g_opencl.k_bias, 2, NULL, global, NULL, 0, NULL, &event);
    register_kernel_time("add_bias", event);
    clReleaseEvent(event);
#else
    clEnqueueNDRangeKernel(g_opencl.queue, g_opencl.k_bias, 2, NULL, global, NULL, 0, NULL, NULL);
#endif
}

void run_gemm(cl_mem A, cl_mem B, cl_mem C, int M, int N, int K, int transB,
    int A_off, int B_off, int C_off, float scale) {
    int TS = 32;
    size_t global[2] = { (size_t)(((K + TS - 1) / TS) * TS), (size_t)(((M + TS - 1) / TS) * TS) };
    size_t local[2] = { (size_t)TS, (size_t)TS };
    int stride = embed_dim;

    clSetKernelArg(g_opencl.k_gemm, 0, sizeof(cl_mem), &A);
    clSetKernelArg(g_opencl.k_gemm, 1, sizeof(cl_mem), &B);
    clSetKernelArg(g_opencl.k_gemm, 2, sizeof(cl_mem), &C);
    clSetKernelArg(g_opencl.k_gemm, 3, sizeof(int), &M);
    clSetKernelArg(g_opencl.k_gemm, 4, sizeof(int), &N);
    clSetKernelArg(g_opencl.k_gemm, 5, sizeof(int), &K);
    clSetKernelArg(g_opencl.k_gemm, 6, sizeof(int), &stride); // A_stride
    clSetKernelArg(g_opencl.k_gemm, 7, sizeof(int), &stride); // B_stride
    clSetKernelArg(g_opencl.k_gemm, 8, sizeof(int), &stride); // C_stride: MHA Score�� ��� �ٸ� ����
    clSetKernelArg(g_opencl.k_gemm, 9, sizeof(int), &A_off);
    clSetKernelArg(g_opencl.k_gemm, 10, sizeof(int), &B_off);
    clSetKernelArg(g_opencl.k_gemm, 11, sizeof(int), &C_off);
    clSetKernelArg(g_opencl.k_gemm, 12, sizeof(int), &transB);
    clSetKernelArg(g_opencl.k_gemm, 13, sizeof(float), &scale);
#ifdef ENABLE_PROFILING
    cl_event event;
    clEnqueueNDRangeKernel(g_opencl.queue, g_opencl.k_gemm, 2, NULL, global, local, 0, NULL, &event);
    register_kernel_time("gemm", event);
    clReleaseEvent(event);
#else
    clEnqueueNDRangeKernel(g_opencl.queue, g_opencl.k_gemm, 2, NULL, global, local, 0, NULL, NULL);
#endif
}

void run_residual(cl_mem in, cl_mem out, int total_elems) {
    size_t global = ((total_elems + 255) / 256) * 256;
    clSetKernelArg(g_opencl.k_res, 0, sizeof(cl_mem), &in);
    clSetKernelArg(g_opencl.k_res, 1, sizeof(cl_mem), &out);
    clSetKernelArg(g_opencl.k_res, 2, sizeof(int), &total_elems);
#ifdef ENABLE_PROFILING
    cl_event event;
    clEnqueueNDRangeKernel(g_opencl.queue, g_opencl.k_res, 1, NULL, &global, NULL, 0, NULL, &event);
    register_kernel_time("residual", event);
    clReleaseEvent(event);
#else
    clEnqueueNDRangeKernel(g_opencl.queue, g_opencl.k_res, 1, NULL, &global, NULL, 0, NULL, NULL);
#endif
}

void run_softmax(cl_mem score, int tokens, int offset) {
    size_t local = 256;
    size_t global = tokens * local;
    clSetKernelArg(g_opencl.k_soft, 0, sizeof(cl_mem), &score);
    clSetKernelArg(g_opencl.k_soft, 1, sizeof(int), &tokens);
    clSetKernelArg(g_opencl.k_soft, 2, sizeof(int), &offset);
#ifdef ENABLE_PROFILING
    cl_event event;
    clEnqueueNDRangeKernel(g_opencl.queue, g_opencl.k_soft, 1, NULL, &global, &local, 0, NULL, &event);
    register_kernel_time("softmax", event);
    clReleaseEvent(event);
#else
    clEnqueueNDRangeKernel(g_opencl.queue, g_opencl.k_res, 1, NULL, &global, NULL, 0, NULL, NULL);
#endif
}

void run_gelu(cl_mem data, int total) {
    size_t global = ((total + 255) / 256) * 256;
    clSetKernelArg(g_opencl.k_gelu, 0, sizeof(cl_mem), &data);
    clSetKernelArg(g_opencl.k_gelu, 1, sizeof(int), &total);
#ifdef ENABLE_PROFILING
    cl_event event;
    clEnqueueNDRangeKernel(g_opencl.queue, g_opencl.k_gelu, 1, NULL, &global, NULL, 0, NULL, &event);
    register_kernel_time("gelu", event);
    clReleaseEvent(event);
#else
    clEnqueueNDRangeKernel(g_opencl.queue, g_opencl.k_gelu, 1, NULL, &global, NULL, 0, NULL, NULL);
#endif
}


void MHA_batched(cl_mem in, cl_mem out, int w_idx, int b_idx, int out_w_idx, int out_b_idx, int batch_size) {
    int tokens = ((img_size / patch_size) * (img_size / patch_size)) + 1;
    int total_tokens = tokens * batch_size;
    int head_dim = embed_dim / num_heads;
    float scale = 1.0f / sqrtf((float)head_dim);

    // --- Q ---
    run_linear(in, g_mem_pool.q_buf, w_idx, b_idx, total_tokens, embed_dim, embed_dim, 0);
    // --- K ---
    run_linear(in, g_mem_pool.k_buf, w_idx, b_idx, total_tokens, embed_dim, embed_dim, embed_dim);
    // --- V ---
    run_linear(in, g_mem_pool.v_buf, w_idx, b_idx, total_tokens, embed_dim, embed_dim, 2 * embed_dim);
    // -----------------------------------------------------
    // Step 2: Calculate Scores (Q * K^T)
    // -----------------------------------------------------
    // Global Size: [Batch * NumHeads, Tokens]
    
    //Tiling o 2d
    //size_t score_global[2] = { (size_t)(batch_size * num_heads), (size_t)tokens };
    //Tiling x 3d
    size_t score_global[3] = { (size_t)(batch_size * num_heads), (size_t)tokens, (size_t)tokens };

    int nh = num_heads;

    clSetKernelArg(g_opencl.k_score, 0, sizeof(cl_mem), &g_mem_pool.q_buf);
    clSetKernelArg(g_opencl.k_score, 1, sizeof(cl_mem), &g_mem_pool.k_buf);
    clSetKernelArg(g_opencl.k_score, 2, sizeof(cl_mem), &g_mem_pool.attn_score);
    clSetKernelArg(g_opencl.k_score, 3, sizeof(int), &tokens);
    clSetKernelArg(g_opencl.k_score, 4, sizeof(int), &head_dim);
    clSetKernelArg(g_opencl.k_score, 5, sizeof(int), &nh);
    clSetKernelArg(g_opencl.k_score, 6, sizeof(float), &scale);

    //clEnqueueNDRangeKernel(g_opencl.queue, g_opencl.k_score, 2, NULL, score_global, NULL, 0, NULL, NULL);
    clEnqueueNDRangeKernel(g_opencl.queue, g_opencl.k_score, 3, NULL, score_global, NULL, 0, NULL, NULL);


    // -----------------------------------------------------
    // Step 3: Softmax
    // -----------------------------------------------------
    // Global Size: [Batch * NumHeads, Tokens] (Score Kernel�� ����)
    clSetKernelArg(g_opencl.k_mha_soft, 0, sizeof(cl_mem), &g_mem_pool.attn_score);
    clSetKernelArg(g_opencl.k_mha_soft, 1, sizeof(int), &tokens);

    clEnqueueNDRangeKernel(g_opencl.queue, g_opencl.k_mha_soft, 2, NULL, score_global, NULL, 0, NULL, NULL);


    // -----------------------------------------------------
    // Step 4: Context (Score * V)
    // -----------------------------------------------------
    // Global Size: [Batch, Tokens, EmbedDim]
    size_t ctx_global[3] = { (size_t)batch_size, (size_t)tokens, (size_t)embed_dim };

    clSetKernelArg(g_opencl.k_context, 0, sizeof(cl_mem), &g_mem_pool.attn_score);
    clSetKernelArg(g_opencl.k_context, 1, sizeof(cl_mem), &g_mem_pool.v_buf);
    clSetKernelArg(g_opencl.k_context, 2, sizeof(cl_mem), &g_mem_pool.attn_out_linear);
    clSetKernelArg(g_opencl.k_context, 3, sizeof(int), &tokens);
    clSetKernelArg(g_opencl.k_context, 4, sizeof(int), &head_dim);
    clSetKernelArg(g_opencl.k_context, 5, sizeof(int), &nh);

    clEnqueueNDRangeKernel(g_opencl.queue, g_opencl.k_context, 3, NULL, ctx_global, NULL, 0, NULL, NULL);


    // -----------------------------------------------------
    // Step 5: Output Projection (Linear)
    // -----------------------------------------------------
    run_linear(g_mem_pool.attn_out_linear, out, out_w_idx, out_b_idx, total_tokens, embed_dim, embed_dim, 0);
}

void MLP_batched(cl_mem in, cl_mem out, int w1, int b1, int w2, int b2, int batch_size) {
    int tokens = ((img_size / patch_size) * (img_size / patch_size)) + 1;
    int total = tokens * batch_size;
    int hidden = (int)(embed_dim * mlp_ratio);

    run_linear(in, g_mem_pool.mlp_fc1_out, w1, b1, total, embed_dim, hidden, 0);
    run_gelu(g_mem_pool.mlp_fc1_out, total * hidden);
    run_linear(g_mem_pool.mlp_fc1_out, out, w2, b2, total, hidden, embed_dim, 0);
}

void Encoder_batched(cl_mem in, cl_mem out, int *net_indices, int batch_size) {
    // Ping Pong or Intermediate
    int tokens = ((img_size / patch_size) * (img_size / patch_size)) + 1;
    int total_elems = tokens * batch_size * embed_dim;

    // LN1
    run_layernorm(in, g_mem_pool.ln_buf, net_indices[0], net_indices[1], tokens * batch_size);
    // MHA
    MHA_batched(g_mem_pool.ln_buf, g_mem_pool.attn_res_buf, net_indices[2], net_indices[3], net_indices[4], net_indices[5], batch_size);
    // Residual 1 (Input + MHA_Out) -> Save to enc_in (inplace) or temp?
    run_residual(g_mem_pool.attn_res_buf, in, total_elems);

    // LN2 (Apply on 'in' which is now residual)
    run_layernorm(in, g_mem_pool.ln_buf, net_indices[6], net_indices[7], tokens * batch_size);
    // MLP
    MLP_batched(g_mem_pool.ln_buf, g_mem_pool.mlp_res_buf, net_indices[8], net_indices[9], net_indices[10], net_indices[11], batch_size);
    // Residual 2
    clEnqueueCopyBuffer(g_opencl.queue, in, out, 0, 0, total_elems * sizeof(float), 0, NULL, NULL);
    run_residual(g_mem_pool.mlp_res_buf, out, total_elems);
}


void ViT_opencl(ImageData *image, Network *networks, float **probabilities) {
    int batch_size = BATCH_SIZE_DEFAULT;
    time_t start, end;
    initialize_opencl();
    init_memory_pool(batch_size);
    load_all_weights(networks);

    int total_images = image[0].n;

    int num_imgs = image[0].n;

    // Host Batch Buffer
    float *h_input = (float *)malloc(sizeof(float) * batch_size * 3 * img_size * img_size);
    float *h_output = (float *)malloc(sizeof(float) * batch_size * num_classes);
    float *cls_token_data = (float *)malloc(sizeof(float) * embed_dim);

    cl_mem logit_buf = clCreateBuffer(g_opencl.context, CL_MEM_READ_WRITE,
        sizeof(float) * batch_size * 197 * num_classes, NULL, NULL);
    cl_mem prob_buf = clCreateBuffer(g_opencl.context, CL_MEM_READ_WRITE,
        sizeof(float) * batch_size * num_classes, NULL, NULL);

    for (int start_idx = 0; start_idx < num_imgs; start_idx += batch_size) {
        int current_batch = (start_idx + batch_size > num_imgs) ? (num_imgs - start_idx) : batch_size;

        // 1. Pack Batch
        for (int b = 0; b < current_batch; b++) {
            memcpy(h_input + b * 3 * img_size * img_size, image[start_idx + b].data, sizeof(float) * 3 * img_size * img_size);
        }
        clEnqueueWriteBuffer(g_opencl.queue, g_mem_pool.batch_input, CL_FALSE, 0,
            current_batch * 3 * img_size * img_size * sizeof(float), h_input, 0, NULL, NULL);

        // 2. Conv2d
        run_conv2d(g_mem_pool.batch_input, g_mem_pool.layer0_out, 1, 2, current_batch);

        // 3. Flatten
        run_flatten(g_mem_pool.layer0_out, g_mem_pool.layer1_out, current_batch);

        // 4. Class Token & Pos Emb
        run_prepare_input(g_mem_pool.layer1_out, g_mem_pool.enc_in, 0, 3, current_batch);

        // 5. Encoder Loop
        cl_mem in_buf = g_mem_pool.enc_in;
        cl_mem out_buf = g_mem_pool.enc_out;

        for (int i = 0; i < 12; i++) {
            int base = 4 + i * 12; // Layer start index
            int indices[12];
            for (int k = 0; k < 12; k++) indices[k] = base + k;

            Encoder_batched(in_buf, out_buf, indices, current_batch);

            // Swap
            cl_mem temp = in_buf; in_buf = out_buf; out_buf = temp;
        }

        // 6. Layer Norm (Final)
        run_layernorm(in_buf, g_mem_pool.enc_out, 148, 149, ((img_size / patch_size) * (img_size / patch_size) + 1) * current_batch);

        // 7. Read Back
        run_linear(g_mem_pool.enc_out, logit_buf, 150, 151, ((img_size / patch_size) * (img_size / patch_size) + 1) * current_batch, embed_dim, num_classes, 0);

        size_t soft_global = current_batch;
        int seq_len = 197;
        int nc = num_classes;

        clSetKernelArg(g_opencl.k_cls_soft, 0, sizeof(cl_mem), &logit_buf);
        clSetKernelArg(g_opencl.k_cls_soft, 1, sizeof(cl_mem), &prob_buf);
        clSetKernelArg(g_opencl.k_cls_soft, 2, sizeof(int), &nc);
        clSetKernelArg(g_opencl.k_cls_soft, 3, sizeof(int), &seq_len);

        clEnqueueNDRangeKernel(g_opencl.queue, g_opencl.k_cls_soft, 1, NULL, &soft_global, NULL, 0, NULL, NULL);

        float *batch_probs = (float *)malloc(sizeof(float) * current_batch * num_classes);

        clEnqueueReadBuffer(g_opencl.queue, prob_buf, CL_TRUE, 0,
            sizeof(float) * current_batch * num_classes, batch_probs, 0, NULL, NULL);

        for (int b = 0; b < current_batch; b++) {
            memcpy(probabilities[start_idx + b], batch_probs + b * num_classes, sizeof(float) * num_classes);
        }

        free(batch_probs);
        printf("%d ~ %d��° �̹��� �Ϸ�\n", start_idx, (start_idx + batch_size) <= 100 ? (start_idx + batch_size) : 100);
    }
    clFinish(g_opencl.queue);
    clReleaseMemObject(prob_buf);
    clReleaseMemObject(logit_buf);
    free(h_input);
    free(h_output);
    free(cls_token_data);
    print_profiling_stats();
    release_resources();
}