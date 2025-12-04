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
#include "ViT_seq.h"
#define img_size 224
#define patch_size 16
#define in_chans 3
#define num_classes 1000
#define embed_dim 768
#define depth 12
#define num_heads 12
#define mlp_ratio 4.0
#define dropout 0.0
#define attn_dropout 0.0
#define drop_path_rate 0.0
#define eps 1e-6

typedef struct {
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_device_id device;
} OpenCL_Resources;

OpenCL_Resources g_opencl;

char *get_source_code(const char *file_name, size_t *len) {
    FILE *file = fopen(file_name, "rb");
    if (file == NULL) {
        printf("[%s:%d] Failed to open %s\n", __FILE__, __LINE__, file_name);
        exit(EXIT_FAILURE);
    }

    fseek(file, 0, SEEK_END);
    size_t length = (size_t)ftell(file);
    rewind(file);

    char *source_code = (char *)malloc(length + 1);
    fread(source_code, length, 1, file);
    source_code[length] = '\0';
    fclose(file);
    *len = length;

    return source_code;
}

void build_error(cl_program program, cl_device_id device, cl_int err) {
    if (err == CL_BUILD_PROGRAM_FAILURE) {
        size_t log_size;
        char *log;

        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        CHECK_ERROR(err);

        log = (char *)malloc(log_size + 1);
        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        CHECK_ERROR(err);

        log[log_size] = '\0';
        printf("Compiler error:\n%s\n", log);
        free(log);
        exit(0);
    };
}

void initialize_opencl() {
    cl_int err;
    cl_platform_id platform;

    err = clGetPlatformIDs(1, &platform, NULL);
    CHECK_ERROR(err);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &g_opencl.device, NULL);
    CHECK_ERROR(err);

    g_opencl.context = clCreateContext(NULL, 1, &g_opencl.device, NULL, NULL, &err);
    CHECK_ERROR(err);

    g_opencl.queue = clCreateCommandQueueWithProperties(g_opencl.context, g_opencl.device, 0, &err);
    CHECK_ERROR(err);

    size_t kernel_source_size;
    char *kernel_source = get_source_code("kernel.cl", &kernel_source_size);
    g_opencl.program = clCreateProgramWithSource(g_opencl.context, 1, (const char **)&kernel_source, &kernel_source_size, &err);
    CHECK_ERROR(err)

    err = clBuildProgram(g_opencl.program, 1, &g_opencl.device, NULL, NULL, NULL);
    build_error(g_opencl.program, g_opencl.device, err);
    CHECK_ERROR(err);

    g_opencl.kernel = clCreateKernel(g_opencl.program, "Conv2d_Kernel", &err);
    CHECK_ERROR(err);
    free(kernel_source);
}

void Release_opencl() {
    cl_int err;
    err = clReleaseKernel(g_opencl.kernel);
    CHECK_ERROR(err);
    err = clReleaseProgram(g_opencl.program);
    CHECK_ERROR(err);
    err = clReleaseCommandQueue(g_opencl.queue);
    CHECK_ERROR(err);
    err = clReleaseContext(g_opencl.context);
    CHECK_ERROR(err);
}

cl_mem create_weight_buffer(Network *network) {
    cl_int err;
    size_t weight_size = network->size * sizeof(float);

    cl_mem weight_buffer = clCreateBuffer(g_opencl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        weight_size, network->data, &err);
    CHECK_ERROR(err);

    return weight_buffer;
}

void Conv2d_opencl(float *input, float *output, Network weight, Network bias) {
    cl_int err;
    const int output_size = img_size / patch_size;
    const size_t input_size = (size_t)in_chans * img_size * img_size * sizeof(float);
    const size_t output_size_bytes = (size_t)embed_dim * output_size * output_size * sizeof(float);

    // --- 0. 메모리 버퍼 생성 ---
    cl_mem input_buf = clCreateBuffer(g_opencl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                      input_size, input, &err);
    CHECK_ERROR(err);
    
    cl_mem weight_buf = create_weight_buffer(&weight); 
    cl_mem bias_buf = create_weight_buffer(&bias); 

    // 최종 결과 버퍼
    cl_mem output_buf = clCreateBuffer(g_opencl.context, CL_MEM_WRITE_ONLY, output_size_bytes, NULL, &err);
    CHECK_ERROR(err);


    // --- 1.Conv2d 커널 실행 ---
    cl_kernel conv2d_integrated_kernel = clCreateKernel(g_opencl.program, "Conv2d_Kernel", &err);
    CHECK_ERROR(err);

    // Global Size: [Output_Channel, Output_Height, Output_Width]
    size_t global_size_conv[3] = { (size_t)embed_dim, (size_t)output_size, (size_t)output_size };

    // 커널 인자 설정
    int is = img_size;
    int ps = patch_size;
    int ic = in_chans;
    int ed = embed_dim;
    clSetKernelArg(conv2d_integrated_kernel, 0, sizeof(cl_mem), &input_buf);
    clSetKernelArg(conv2d_integrated_kernel, 1, sizeof(cl_mem), &output_buf);
    clSetKernelArg(conv2d_integrated_kernel, 2, sizeof(cl_mem), &weight_buf);
    clSetKernelArg(conv2d_integrated_kernel, 3, sizeof(cl_mem), &bias_buf);
    clSetKernelArg(conv2d_integrated_kernel, 4, sizeof(int), &is);
    clSetKernelArg(conv2d_integrated_kernel, 5, sizeof(int), &ps);
    clSetKernelArg(conv2d_integrated_kernel, 6, sizeof(int), &ic);
    clSetKernelArg(conv2d_integrated_kernel, 7, sizeof(int), &ed);
    clSetKernelArg(conv2d_integrated_kernel, 8, sizeof(int), &output_size); // Output Size 추가

    err = clEnqueueNDRangeKernel(g_opencl.queue, conv2d_integrated_kernel, 3, NULL, global_size_conv, NULL, 0, NULL, NULL);
    CHECK_ERROR(err);
    clReleaseKernel(conv2d_integrated_kernel);


    // --- 2. 결과 검색 및 자원 해제 ---
    clFinish(g_opencl.queue);
    err = clEnqueueReadBuffer(g_opencl.queue, output_buf, CL_TRUE, 0, output_size_bytes, output, 0, NULL, NULL);
    CHECK_ERROR(err);

    clReleaseMemObject(input_buf);
    clReleaseMemObject(weight_buf);
    clReleaseMemObject(bias_buf);
    clReleaseMemObject(output_buf);
}

void flatten_transpose_opencl(float *input, float *output) {
    cl_int err;

    // 상수 계산 (외부 #define 값 사용)
    const int output_size = img_size / patch_size;
    const int num_patches = output_size * output_size; // 196
    const size_t input_size_bytes = (size_t)embed_dim * output_size * output_size * sizeof(float);
    const size_t output_size_bytes = (size_t)num_patches * embed_dim * sizeof(float);

    // --- 1. 메모리 버퍼 생성 ---

    // 입력 버퍼 (Conv2d 결과)
    cl_mem input_buf = clCreateBuffer(g_opencl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        input_size_bytes, input, &err);
    CHECK_ERROR(err);

    // 출력 버퍼 (Flattened, Transposed 결과)
    cl_mem output_buf = clCreateBuffer(g_opencl.context, CL_MEM_WRITE_ONLY, output_size_bytes, NULL, &err);
    CHECK_ERROR(err);

    // --- 2. 커널 실행 ---
    cl_kernel flat_transpose_kernel = clCreateKernel(g_opencl.program, "FlattenTranspose_Kernel", &err);
    CHECK_ERROR(err);

    // Global Size: [num_patches, embed_dim] (196 x 768)
    // Work-Item 총 개수: 150,528개
    size_t global_size_flat[2] = { (size_t)num_patches, (size_t)embed_dim };

    // 커널 인자 설정
    int ed = embed_dim;
    clSetKernelArg(flat_transpose_kernel, 0, sizeof(cl_mem), &input_buf);
    clSetKernelArg(flat_transpose_kernel, 1, sizeof(cl_mem), &output_buf);
    clSetKernelArg(flat_transpose_kernel, 2, sizeof(int), &output_size);
    clSetKernelArg(flat_transpose_kernel, 3, sizeof(int), &ed);

    // 커널 실행
    err = clEnqueueNDRangeKernel(g_opencl.queue, flat_transpose_kernel, 2, NULL, global_size_flat, NULL, 0, NULL, NULL);
    CHECK_ERROR(err);
    clReleaseKernel(flat_transpose_kernel);

    // --- 3. 결과 검색 및 자원 해제 ---
    clFinish(g_opencl.queue);

    // GPU 결과 -> Host output으로 복사
    err = clEnqueueReadBuffer(g_opencl.queue, output_buf, CL_TRUE, 0, output_size_bytes, output, 0, NULL, NULL);
    CHECK_ERROR(err);

    clReleaseMemObject(input_buf);
    clReleaseMemObject(output_buf);
}

void layer_norm_opencl(float *input, float *output, Network weight, Network bias)
{
    int token = ((img_size / patch_size) * (img_size / patch_size)) + 1;
    cl_int err;

    cl_context context = g_opencl.context;
    cl_command_queue queue = g_opencl.queue;
    cl_program program = g_opencl.program;
    g_opencl.kernel = clCreateKernel(g_opencl.program, "layer_norm_kernel", &err);
    cl_kernel kernel = g_opencl.kernel;

    // buffer
    cl_mem input_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * token * embed_dim, input, &err);    //input buf
    CHECK_ERROR(err);
    cl_mem output_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * token * embed_dim, NULL, &err);                          //output buf
    CHECK_ERROR(err);
    cl_mem weight_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * embed_dim, weight.data, &err);     //weight buf
    CHECK_ERROR(err);
    cl_mem bias_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * embed_dim, bias.data, &err);         //bias buf
    CHECK_ERROR(err);

    // work group size
    const size_t local_size = 256;                      // local
    const size_t global_size = token * local_size;      // global(local * token)

    size_t local_mem_size = (local_size * 2 + 2) * sizeof(float);   // E(x), E(x^2)의 분자 값을 reduction할 메모리(*2), 최종 E(x), E(x^2)도 저장할 공간 (+2)

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buf);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buf);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &weight_buf);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &bias_buf);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 4, local_mem_size, NULL);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 5, sizeof(int), &token);
    CHECK_ERROR(err);
    int global_offset = 0;
    err = clSetKernelArg(kernel, 6, sizeof(int), &global_offset);
    CHECK_ERROR(err);

    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
    CHECK_ERROR(err);
    clFinish(queue);

    start = clock();
    err = clEnqueueReadBuffer(queue, output_buf, CL_TRUE, 0, sizeof(float) * token * embed_dim, output, 0, NULL, NULL);
    CHECK_ERROR(err);


    err = clReleaseMemObject(input_buf);
    err = clReleaseMemObject(output_buf);
    err = clReleaseMemObject(weight_buf);
    err = clReleaseMemObject(bias_buf);

    CHECK_ERROR(err);
}

void gemm_ScoresV(float *scores_host, float *V_host, float *attn_output_host, int head_offset) {
    cl_int err;
    int TS = 2;
    // 상수 계산
    const int tokens = ((img_size / patch_size) * (img_size / patch_size)) + 1;
    const int head_dim = embed_dim / num_heads;

    const size_t score_matrix_size = (size_t)tokens * tokens * sizeof(float);
    const size_t Vhead_matrix_size = (size_t)tokens * head_dim * sizeof(float);
    const size_t head_out_matrix_size = (size_t)tokens * head_dim * sizeof(float);

    // 1. Host 데이터를 GPU 버퍼로 전송

    // scores (A) 버퍼: [tokens x tokens]
    cl_mem scores_buf = clCreateBuffer(g_opencl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        score_matrix_size, scores_host, &err);
    CHECK_ERROR(err);

    // Vhead (B) 버퍼: [tokens x head_dim]. V 전체에서 해당 head 부분만 추출
    float *Vhead_tmp = (float *)malloc(Vhead_matrix_size);
    if (!Vhead_tmp) { fprintf(stderr, "malloc failed\n"); exit(EXIT_FAILURE); }
    for (int i = 0; i < tokens; ++i) {
        // V[i * embed_dim + head_offset]에서 head_dim 크기만큼 복사
        memcpy(Vhead_tmp + i * head_dim, V_host + i * embed_dim + head_offset, head_dim * sizeof(float));
    }

    cl_mem Vhead_buf = clCreateBuffer(g_opencl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        Vhead_matrix_size, Vhead_tmp, &err);
    CHECK_ERROR(err);
    free(Vhead_tmp);

    // head_out (C) 버퍼: [tokens x head_dim]
    cl_mem head_out_buf = clCreateBuffer(g_opencl.context, CL_MEM_WRITE_ONLY, head_out_matrix_size, NULL, &err);
    CHECK_ERROR(err);

    // 2. 커널 생성 및 인자 설정
    cl_kernel gemm_kernel = clCreateKernel(g_opencl.program, "gemm", &err);
    CHECK_ERROR(err);

    const int ROW_A = tokens;
    const int COL_A = tokens;
    const int ROW_B = tokens;
    const int COL_B = head_dim;


    clSetKernelArg(gemm_kernel, 0, sizeof(cl_mem), &scores_buf); // A
    clSetKernelArg(gemm_kernel, 1, sizeof(cl_mem), &Vhead_buf);  // B
    clSetKernelArg(gemm_kernel, 2, sizeof(cl_mem), &head_out_buf); // C
    clSetKernelArg(gemm_kernel, 3, sizeof(int), &ROW_A);
    clSetKernelArg(gemm_kernel, 4, sizeof(int), &COL_A);
    clSetKernelArg(gemm_kernel, 5, sizeof(int), &ROW_B);
    clSetKernelArg(gemm_kernel, 6, sizeof(int), &COL_B);

    // 3. 커널 실행 
    // Global Size: C[ROW_A x COL_B]
    size_t global_size_gemm[2] = { (size_t)COL_B, (size_t)ROW_A };
    size_t local_size_gemm[2] = { TS, TS };

    // Global Size를 TS의 배수로 올림
    global_size_gemm[0] = ((global_size_gemm[0] + TS - 1) / TS) * TS;
    global_size_gemm[1] = ((global_size_gemm[1] + TS - 1) / TS) * TS;

    err = clEnqueueNDRangeKernel(g_opencl.queue, gemm_kernel, 2, NULL, global_size_gemm, local_size_gemm, 0, NULL, NULL);
    CHECK_ERROR(err);


    float *head_out_host = (float *)malloc(head_out_matrix_size);
    if (!head_out_host) { fprintf(stderr, "malloc failed\n"); exit(EXIT_FAILURE); }

    err = clEnqueueReadBuffer(g_opencl.queue, head_out_buf, CL_TRUE, 0, head_out_matrix_size, head_out_host, 0, NULL, NULL);
    CHECK_ERROR(err);

    // 5. head_out 결과를 최종 attn_output 버퍼에 복사 (C 호스트 로직)
    for (int i = 0; i < tokens; i++) {
        memcpy(attn_output_host + i * embed_dim + head_offset, head_out_host + i * head_dim, head_dim * sizeof(float));
    }

    // 6. 자원 해제
    free(head_out_host);
    clReleaseMemObject(scores_buf);
    clReleaseMemObject(Vhead_buf);
    clReleaseMemObject(head_out_buf);
    clReleaseKernel(gemm_kernel);
}

// ----------------------------------------------------
// MHA GEMM 커널 실행 헬퍼 함수
// ----------------------------------------------------
cl_int enqueue_gemm_stage(cl_command_queue queue, cl_kernel kernel,
    cl_mem A, cl_mem B, cl_mem C,
    int M, int N, int K,
    int A_row_stride, int B_row_stride, int C_row_stride,
    int A_offset, int B_offset, int C_offset,
    int transpose_B, float scale_factor,
    cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
    cl_event *event_out)
{
    cl_int err;
    int TS = 16;
    size_t M_padded = ((M + TS - 1) / TS) * TS; // M = 197 -> 198
    size_t K_padded = ((K + TS - 1) / TS) * TS; // K = 197 -> 198

    // Global Work Size 설정
    size_t global_work_size[2] = { K_padded, M_padded };
    size_t local_work_size[2] = { (size_t)TS, (size_t)TS }; //

    // 커널 인자 설정
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &A);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &B);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &C);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &M);
    err |= clSetKernelArg(kernel, 4, sizeof(int), &N);
    err |= clSetKernelArg(kernel, 5, sizeof(int), &K);
    err |= clSetKernelArg(kernel, 6, sizeof(int), &A_row_stride);
    err |= clSetKernelArg(kernel, 7, sizeof(int), &B_row_stride);
    err |= clSetKernelArg(kernel, 8, sizeof(int), &C_row_stride);
    err |= clSetKernelArg(kernel, 9, sizeof(int), &A_offset);
    err |= clSetKernelArg(kernel, 10, sizeof(int), &B_offset);
    err |= clSetKernelArg(kernel, 11, sizeof(int), &C_offset);
    err |= clSetKernelArg(kernel, 12, sizeof(int), &transpose_B);
    err |= clSetKernelArg(kernel, 13, sizeof(float), &scale_factor);
    CHECK_ERROR(err);

    // 커널 실행
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL,
        global_work_size, local_work_size,
        num_events_in_wait_list, event_wait_list, event_out);
    CHECK_ERROR(err);
    return err;
}


// ----------------------------------------------------
// Softmax 커널 실행 헬퍼 함수
// ----------------------------------------------------
cl_int enqueue_softmax_stage(cl_command_queue queue, cl_kernel kernel,
    cl_mem scores, int tokens, int score_offset,
    cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
    cl_event *event_out)
{
    cl_int err;

    // 하나의 WorkGroup(256 threads)이 하나의 Token Row를 처리하도록 설정
    // Global Work Size = (처리할 행 개수) * (그룹 당 스레드 수)
    size_t local_size = 256;
    size_t global_size = tokens * local_size;

    size_t global_work_size[1] = { global_size };
    size_t local_work_size[1] = { local_size };

    // 커널 인자 설정
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &scores);
    err |= clSetKernelArg(kernel, 1, sizeof(int), &tokens);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &score_offset);
    //err |= clSetKernelArg(kernel, 3, local_mem_size, NULL); // 로컬 메모리 전달
    CHECK_ERROR(err);

    // 커널 실행
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL,
        global_work_size, local_work_size,
        num_events_in_wait_list, event_wait_list, event_out);
    CHECK_ERROR(err);
    return err;
}



void multihead_attn_opencl(float *input, float *output,
    Network in_weight, Network in_bias, Network out_weight, Network out_bias)
{
    cl_int err;
    int head_dim = embed_dim / num_heads;
    int tokens = ((img_size / patch_size) * (img_size / patch_size)) + 1;
    int N_heads = num_heads;
    int Embed_dim = embed_dim;

    // GEMM 커널의 Local Work Size (TS=16 가정)
    const int TS = 4;

    /* 1. Host Memory Allocation & QKV Calculation (CPU) */
    int Q_dim = 0, K_dim = embed_dim, V_dim = embed_dim * 2;
    float *Q = (float *)malloc(sizeof(float) * tokens * embed_dim);
    float *K = (float *)malloc(sizeof(float) * tokens * embed_dim);
    float *V = (float *)malloc(sizeof(float) * tokens * embed_dim);

    // Q, K, V 계산 (원본 C 코드 로직 유지)
    for (int t = 0; t < tokens; t++) {
        float sum_q, sum_k, sum_v;
        for (int i = 0; i < embed_dim; i++) {
            sum_q = in_bias.data[Q_dim + i], sum_k = in_bias.data[K_dim + i], sum_v = in_bias.data[V_dim + i];
            for (int j = 0; j < embed_dim; j++) {
                sum_q += input[t * embed_dim + j] * in_weight.data[(Q_dim + i) * embed_dim + j];
                sum_k += input[t * embed_dim + j] * in_weight.data[(K_dim + i) * embed_dim + j];
                sum_v += input[t * embed_dim + j] * in_weight.data[(V_dim + i) * embed_dim + j];
            }
            Q[t * embed_dim + i] = sum_q;
            K[t * embed_dim + i] = sum_k;
            V[t * embed_dim + i] = sum_v;
        }
    }

    float *attn_output = (float *)malloc(sizeof(float) * tokens * embed_dim);
    size_t scores_size = sizeof(float) * tokens * tokens * num_heads;

    cl_mem Q_d = clCreateBuffer(g_opencl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * tokens * embed_dim, Q, &err);
    CHECK_ERROR(err);
    cl_mem K_d = clCreateBuffer(g_opencl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * tokens * embed_dim, K, &err);
    CHECK_ERROR(err);
    cl_mem V_d = clCreateBuffer(g_opencl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * tokens * embed_dim, V, &err);
    CHECK_ERROR(err);

    cl_mem scores_d = clCreateBuffer(g_opencl.context, CL_MEM_READ_WRITE, scores_size, NULL, &err);
    CHECK_ERROR(err);
    cl_mem attn_output_d = clCreateBuffer(g_opencl.context, CL_MEM_READ_WRITE, sizeof(float) * tokens * embed_dim, NULL, &err);
    CHECK_ERROR(err);

    float scale_factor = 1.0f / sqrtf((float)head_dim);

    cl_kernel kernel_gemm = clCreateKernel(g_opencl.program, "MHA_gemm_kernel", &err);
    CHECK_ERROR(err);
    cl_kernel kernel_softmax = clCreateKernel(g_opencl.program, "softmax_reduction_kernel", &err);
    CHECK_ERROR(err);

    cl_event head_completion_events[num_heads];

    for (int h = 0; h < num_heads; h++) {
        size_t head_offset = h * head_dim;
        size_t score_offset = h * tokens * tokens;

        cl_event score_event, softmax_event;

        enqueue_gemm_stage(g_opencl.queue, kernel_gemm, Q_d, K_d, scores_d,
            tokens, head_dim, tokens, embed_dim, embed_dim, tokens,
            head_offset, head_offset, score_offset, 1, scale_factor,
            0, NULL, &score_event);
        enqueue_softmax_stage(g_opencl.queue, kernel_softmax, scores_d,
            tokens, score_offset, 1, &score_event, &softmax_event);
        clReleaseEvent(score_event);
        enqueue_gemm_stage(g_opencl.queue, kernel_gemm, scores_d, V_d, attn_output_d,
            tokens, tokens, head_dim, tokens, embed_dim, embed_dim,
            score_offset, head_offset, head_offset, 0, 1.0f,
            1, &softmax_event, &head_completion_events[h]);
        clReleaseEvent(softmax_event);
    }

    cl_event read_event;

    clEnqueueReadBuffer(g_opencl.queue, attn_output_d, CL_FALSE, 0,
        sizeof(float) * tokens * embed_dim, attn_output,
        num_heads, head_completion_events, &read_event);

    clWaitForEvents(1, &read_event);
    clReleaseEvent(read_event);

    for (int h = 0; h < num_heads; h++) {
        clReleaseEvent(head_completion_events[h]);
    }

    free(Q); free(K); free(V);

    // 최종 선형 프로젝션
    for (int t = 0; t < tokens; t++) {
        for (int i = 0; i < embed_dim; i++) {
            float sum = out_bias.data[i];
            for (int j = 0; j < embed_dim; j++) {
                sum += attn_output[t * embed_dim + j] * out_weight.data[i * embed_dim + j];
            }
            output[t * embed_dim + i] = sum;
        }
    }

    free(attn_output);
    clReleaseKernel(kernel_gemm);
    clReleaseKernel(kernel_softmax);
    clReleaseMemObject(Q_d); clReleaseMemObject(K_d); clReleaseMemObject(V_d);
    clReleaseMemObject(scores_d); clReleaseMemObject(attn_output_d);
}

void compare_buffers(const char *stage_name, const float *buf_cpu, const float *buf_opencl, size_t num_elements) {
    float max_abs_error = 0.0f;
    float max_cpu_val = 0.0f;
    float max_ocl_val = 0.0f;
    int max_error_idx = -1;
    int mismatch_count = 0;
    const float threshold = 1e-4f;

    for (size_t i = 0; i < num_elements; i++) {
        float diff = fabsf(buf_cpu[i] - buf_opencl[i]);

        if (fabsf(buf_cpu[i]) > max_cpu_val) max_cpu_val = fabsf(buf_cpu[i]);
        if (fabsf(buf_opencl[i]) > max_ocl_val) max_ocl_val = fabsf(buf_opencl[i]);

        if (diff > max_abs_error) {
            max_abs_error = diff;
            max_error_idx = i;
        }

        if (diff > threshold) {
            mismatch_count++;
        }
    }

    printf("\n--- Stage Comparison: %s ---\n", stage_name);
    printf("Total Elements: %zu, Mismatched: %d\n", num_elements, mismatch_count);
    printf("Max Abs Error: %e (Threshold: %e)\n", max_abs_error, threshold);

    if (max_abs_error > threshold) {
        printf("ERROR LOCATED HERE! Max CPU Val: %f, Max OpenCL Val: %f\n", max_cpu_val, max_ocl_val);
        printf("Index %d: CPU = %f, OpenCL = %f, Diff = %e\n",
            max_error_idx, buf_cpu[max_error_idx], buf_opencl[max_error_idx], max_abs_error);
    }
    else {
        printf("OK.Results match(FPE expected).\n");
    }
}

void compare_mha_outputs(float *input, Network in_weight, Network in_bias, Network out_weight, Network out_bias) {

    // 1. 메모리 할당
    int tokens = ((img_size / patch_size) * (img_size / patch_size)) + 1;
    size_t output_size = sizeof(float) * tokens * embed_dim;

    float *output_cpu = (float *)malloc(output_size);
    float *output_opencl = (float *)malloc(output_size);

    // 최종 출력 결과를 비교하려면, MHA의 최종 선형 프로젝션까지 포함하는
    // 전체 블록의 출력 버퍼(output)를 사용해야 합니다.

    if (!output_cpu || !output_opencl) {
        fprintf(stderr, "Error: Failed to allocate memory for comparison.\n");
        exit(1);
    }

    printf("--- MHA 출력 비교 시작 (Tokens: %d, Embed Dim: %d) ---\n", tokens, embed_dim);

    // 2. 순차 실행 (CPU)
    printf("1. Running Sequential MHA...\n");
    multihead_attn(input, output_cpu, in_weight, in_bias, out_weight, out_bias);

    // 3. 병렬 실행 (OpenCL)
    printf("2. Running Parallel MHA (OpenCL)...\n");
    multihead_attn_opencl(input, output_opencl, in_weight, in_bias, out_weight, out_bias);

    // 4. 결과 비교
    float max_abs_error = 0.0f;
    int max_error_idx = -1;
    int mismatch_count = 0;
    const float threshold = 1e-4f; // 허용 오차 (부동 소수점 비교용)

    for (int i = 0; i < tokens * embed_dim; i++) {
        float diff = fabsf(output_cpu[i] - output_opencl[i]);

        if (diff > max_abs_error) {
            max_abs_error = diff;
            max_error_idx = i;
        }

        if (diff > threshold) {
            mismatch_count++;
        }
    }

    // 5. 결과 출력
    printf("--- Comparison Results ---\n");
    printf("Total elements: %zu\n", tokens * embed_dim);
    printf("Max Absolute Error: %e\n", max_abs_error);

    if (max_abs_error > threshold) {
        printf("Discrepancy Found! (Mismatch Count: %d)\n", mismatch_count);
        printf("Index with Max Error (%d): CPU = %f, OpenCL = %f\n",
            max_error_idx, output_cpu[max_error_idx], output_opencl[max_error_idx]);

        // 오차가 큰 경우, 디버깅을 위해 몇 가지 값 출력
        if (mismatch_count > 0) {
            printf("\n--- First 5 Mismatched Values ---\n");
            int count = 0;
            for (int i = 0; i < tokens * embed_dim && count < 5; i++) {
                float diff = fabsf(output_cpu[i] - output_opencl[i]);
                if (diff > threshold) {
                    printf("Idx %d (Row %d, Dim %d): CPU: %.6f, OpenCL: %.6f, Diff: %.6e\n",
                        i, i / embed_dim, i % embed_dim, output_cpu[i], output_opencl[i], diff);
                    count++;
                }
            }
        }

    }
    else {
        printf("Results Match within threshold (%.6e).\n", threshold);
    }

    // 메모리 해제
    free(output_cpu);
    free(output_opencl);

    printf("----------------------------------------\n");
}

void mha_debug_compare_stages(float *input, Network in_weight, Network in_bias, Network out_weight, Network out_bias)
{
    cl_int err;
    int head_dim = embed_dim / num_heads;
    int tokens = ((img_size / patch_size) * (img_size / patch_size)) + 1;
    const int TS = 4; // MHA_gemm_kernel의 TS와 일치해야 함

    // 1. QKV Calculation (CPU)
    int Q_dim = 0, K_dim = embed_dim, V_dim = embed_dim * 2;
    size_t qkv_size_bytes = sizeof(float) * tokens * embed_dim;
    size_t qkv_size_elements = tokens * embed_dim;

    // CPU QKV 버퍼
    float *Q_cpu = (float *)malloc(qkv_size_bytes);
    float *K_cpu = (float *)malloc(qkv_size_bytes);
    float *V_cpu = (float *)malloc(qkv_size_bytes);

    // Q, K, V 계산 (호스트 CPU에서 공통으로 수행)
    for (int t = 0; t < tokens; t++) {
        float sum_q, sum_k, sum_v;
        for (int i = 0; i < embed_dim; i++) {
            sum_q = in_bias.data[Q_dim + i], sum_k = in_bias.data[K_dim + i], sum_v = in_bias.data[V_dim + i];
            for (int j = 0; j < embed_dim; j++) {
                sum_q += input[t * embed_dim + j] * in_weight.data[(Q_dim + i) * embed_dim + j];
                sum_k += input[t * embed_dim + j] * in_weight.data[(K_dim + i) * embed_dim + j];
                sum_v += input[t * embed_dim + j] * in_weight.data[(V_dim + i) * embed_dim + j];
            }
            Q_cpu[t * embed_dim + i] = sum_q;
            K_cpu[t * embed_dim + i] = sum_k;
            V_cpu[t * embed_dim + i] = sum_v;
        }
    }

    // OpenCL 버퍼 크기
    size_t scores_num_elements_per_head = tokens * tokens;
    size_t scores_total_elements = scores_num_elements_per_head * num_heads;
    size_t final_attn_output_size = sizeof(float) * qkv_size_elements;

    // OpenCL 버퍼 생성 (QKV는 복사)
    cl_mem Q_d = clCreateBuffer(g_opencl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, qkv_size_bytes, Q_cpu, &err); CHECK_ERROR(err);
    cl_mem K_d = clCreateBuffer(g_opencl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, qkv_size_bytes, K_cpu, &err); CHECK_ERROR(err);
    cl_mem V_d = clCreateBuffer(g_opencl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, qkv_size_bytes, V_cpu, &err); CHECK_ERROR(err);

    // Scores 버퍼 (num_heads * tokens * tokens)
    cl_mem scores_d = clCreateBuffer(g_opencl.context, CL_MEM_READ_WRITE, scores_total_elements * sizeof(float), NULL, &err); CHECK_ERROR(err);
    // Attn Output 버퍼
    cl_mem attn_output_d = clCreateBuffer(g_opencl.context, CL_MEM_READ_WRITE, final_attn_output_size, NULL, &err); CHECK_ERROR(err);

    float scale_factor = 1.0f / sqrtf((float)head_dim);

    cl_kernel kernel_gemm = clCreateKernel(g_opencl.program, "MHA_gemm_kernel", &err); CHECK_ERROR(err);
    cl_kernel kernel_softmax = clCreateKernel(g_opencl.program, "softmax_reduction_kernel", &err); CHECK_ERROR(err);


    // ----------------------------------------------------
    // Intermediate Buffers for Comparison
    // ----------------------------------------------------
    // Raw Scores (QK^T)
    float *scores_cpu_raw = (float *)malloc(scores_total_elements * sizeof(float));
    float *scores_opencl_raw = (float *)malloc(scores_total_elements * sizeof(float));

    // Softmaxed Scores 
    float *scores_cpu_softmaxed = (float *)malloc(scores_total_elements * sizeof(float));
    float *scores_opencl_read = (float *)malloc(scores_total_elements * sizeof(float)); // OpenCL Softmax 결과 (순차 코드 Softmaxed Scores와 비교용)

    // Final Attn Head Output
    float *attn_output_cpu_ref = (float *)malloc(final_attn_output_size);
    float *attn_output_opencl_read = (float *)malloc(final_attn_output_size);


    // ----------------------------------------------------
    // Loop for Head-wise Execution
    // ----------------------------------------------------
    cl_event raw_scores_gemm_events[num_heads]; // Raw Scores GEMM 완료 이벤트
    cl_event final_completion_events[num_heads]; // 최종 출력 이벤트

    for (int h = 0; h < num_heads; h++) {
        int head_offset = h * head_dim;
        int score_offset = h * scores_num_elements_per_head;
        size_t score_offset_bytes = score_offset * sizeof(float);

        cl_event score_gemm_event, write_softmax_event, final_gemm_event;

        /* --- 1. CPU: Q * K^T (Raw Score Calculation) --- */
        for (int i = 0; i < tokens; i++) {
            size_t row_offset = score_offset + i * tokens;
            for (int j = 0; j < tokens; j++) {
                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    float q = Q_cpu[i * embed_dim + head_offset + d];
                    float k = K_cpu[j * embed_dim + head_offset + d];
                    score += q * k;
                }
                scores_cpu_raw[row_offset + j] = score / sqrtf((float)head_dim);
            }
        }

        /* --- 2. OpenCL: Q * K^T -> Scores_d (Raw Scores) --- */
        enqueue_gemm_stage(g_opencl.queue, kernel_gemm, Q_d, K_d, scores_d,
            tokens, head_dim, tokens, embed_dim, embed_dim, tokens,
            head_offset, head_offset, score_offset, 1, scale_factor,
            0, NULL, &score_gemm_event);

        raw_scores_gemm_events[h] = score_gemm_event; // Raw Scores GEMM 완료 이벤트 저장


        /* --- 3. Raw Scores Readback (OpenCL -> CPU) --- */
        // Raw Scores GEMM 완료를 기다림 (Stage 2 비교를 위해)
        clWaitForEvents(1, &score_gemm_event);
        clEnqueueReadBuffer(g_opencl.queue, scores_d, CL_TRUE, score_offset_bytes,
            scores_num_elements_per_head * sizeof(float), &scores_opencl_raw[score_offset],
            0, NULL, NULL);


        /* --- 4. CPU: Softmax Injection (OpenCL Raw Scores -> Softmaxed Scores) --- */
        // OpenCL Raw Scores (scores_opencl_raw)에 순차 Softmax를 적용하여 scores_cpu_softmaxed에 저장
        for (int i = 0; i < tokens; i++) {
            size_t row_offset = score_offset + i * tokens;
            float max_val;

            // Max Reduction
            max_val = scores_opencl_raw[row_offset]; // OpenCL Raw Scores 사용
            for (int j = 1; j < tokens; j++) {
                if (scores_opencl_raw[row_offset + j] > max_val) max_val = scores_opencl_raw[row_offset + j];
            }

            float sum_exp = 0.0f;
            // Exp and Sum Reduction
            for (int j = 0; j < tokens; j++) {
                // CPU expf 사용
                float exp_val = expf(scores_opencl_raw[row_offset + j] - max_val);
                scores_cpu_softmaxed[row_offset + j] = exp_val; // Softmaxed 결과 저장
                sum_exp += exp_val;
            }

            // Normalization
            for (int j = 0; j < tokens; j++) {
                scores_cpu_softmaxed[row_offset + j] /= sum_exp;
            }
        }

        /* --- 5. Softmaxed Scores Writeback (CPU -> OpenCL) --- */
        // CPU에서 계산된 Softmaxed Scores를 OpenCL scores_d 버퍼에 다시 씁니다.
        clEnqueueWriteBuffer(g_opencl.queue, scores_d, CL_TRUE, score_offset_bytes,
            scores_num_elements_per_head * sizeof(float), &scores_cpu_softmaxed[score_offset],
            0, NULL, &write_softmax_event);


        /* --- 6. CPU: Final Output Reference (Softmax Scores * V) --- */
        for (int i = 0; i < tokens; i++) {
            for (int d = 0; d < head_dim; d++) {
                float sum = 0.0f;
                for (int j = 0; j < tokens; j++) {
                    // CPU의 Softmax 결과를 사용하여 최종 출력 레퍼런스 계산
                    sum += scores_cpu_softmaxed[score_offset + i * tokens + j] * V_cpu[j * embed_dim + head_offset + d];
                }
                attn_output_cpu_ref[i * embed_dim + head_offset + d] = sum;
            }
        }

        /* --- 7. OpenCL: Scores * V -> Attn_Output --- */
        // OpenCL Scores_d 버퍼는 이제 CPU에서 계산된 Softmaxed Scores를 포함합니다.
        enqueue_gemm_stage(g_opencl.queue, kernel_gemm, scores_d, V_d, attn_output_d,
            tokens, tokens, head_dim, tokens, embed_dim, embed_dim,
            score_offset, head_offset, head_offset, 0, 1.0f,
            1, &write_softmax_event, &final_gemm_event);
        clReleaseEvent(write_softmax_event);

        final_completion_events[h] = final_gemm_event;
    }


    // ----------------------------------------------------
    // Final Readback and Comparison
    // ----------------------------------------------------

    // Stage 4: Final Attn Output Readback
    clWaitForEvents(num_heads, final_completion_events); // 모든 최종 GEMM 완료 대기
    cl_event read_attn_event;
    err = clEnqueueReadBuffer(g_opencl.queue, attn_output_d, CL_TRUE, 0,
        final_attn_output_size, attn_output_opencl_read,
        num_heads, final_completion_events, &read_attn_event); CHECK_ERROR(err);
    clWaitForEvents(1, &read_attn_event);
    clReleaseEvent(read_attn_event);


    // ----------------------------------------------------
    // Comparison
    // ----------------------------------------------------
    printf("\n\n=============== STAGE-WISE DEBUGGING RESULTS (V4 - HYBRID) ===============");

    // Stage 1: QKV Check (Placebo check)
    compare_buffers("1. QKV Input Check (Q Matrix)", Q_cpu, Q_cpu, qkv_size_elements);

    // Stage 2: Raw Scores Check (Q*K^T GEMM Output Check)
    compare_buffers("2. Raw Scores Check (Q*K^T GEMM)", scores_cpu_raw, scores_opencl_raw, scores_total_elements);

    // Stage 3: Softmaxed Scores Check (CPU Softmax vs. CPU Ref Softmaxed Scores)
    // CPU Softmaxed Scores (scores_cpu_softmaxed)는 OpenCL Raw Scores를 기반으로 계산되었으므로,
    // Stage 2가 OK라면 Stage 3는 FPE 수준의 오차만 보여야 합니다.
    compare_buffers("3. Softmaxed Scores (CPU Softmax Injection)", scores_cpu_softmaxed, scores_cpu_softmaxed, scores_total_elements); // Placebo check

    // Stage 4: Final Attn Head Output Check
    compare_buffers("4. Final Attn Head Output (Scores * V)", attn_output_cpu_ref, attn_output_opencl_read, qkv_size_elements);

    // ----------------------------------------------------
    // Cleanup
    // ----------------------------------------------------
    free(Q_cpu); free(K_cpu); free(V_cpu);
    free(scores_cpu_raw);
    free(scores_opencl_raw);
    free(scores_cpu_softmaxed);
    free(scores_opencl_read);
    free(attn_output_cpu_ref);
    free(attn_output_opencl_read);

    // Event Cleanup
    for (int h = 0; h < num_heads; h++) {
        clReleaseEvent(final_completion_events[h]);
    }
    clReleaseKernel(kernel_gemm);
    clReleaseKernel(kernel_softmax);
    clReleaseMemObject(Q_d); clReleaseMemObject(K_d); clReleaseMemObject(V_d);
    clReleaseMemObject(scores_d); clReleaseMemObject(attn_output_d);
}
////////////////////////////////////// Encoder Architecture //////////////////////////////////////
void Encoder_opencl(float *input, float *output,
    Network ln1_w, Network ln1_b, Network attn_w, Network attn_b, Network attn_out_w, Network attn_out_b,
    Network ln2_w, Network ln2_b, Network mlp1_w, Network mlp1_b, Network mlp2_w, Network mlp2_b) {
    int tokens = ((img_size / patch_size) * (img_size / patch_size)) + 1;
    float *ln1_out = (float *)malloc(sizeof(float) * tokens * embed_dim);
    float *attn_out = (float *)malloc(sizeof(float) * tokens * embed_dim);
    float *residual = (float *)malloc(sizeof(float) * tokens * embed_dim);
    float *ln2_out = (float *)malloc(sizeof(float) * tokens * embed_dim);
    float *mlp_out = (float *)malloc(sizeof(float) * tokens * embed_dim);

    /*LN1*/
    layer_norm_opencl(input, ln1_out, ln1_w, ln1_b);

    /*Attn*/
    multihead_attn_opencl(ln1_out, attn_out, attn_w, attn_b, attn_out_w, attn_out_b);

    /*Residual1*/
    for (int i = 0; i < tokens * embed_dim; i++) {
        residual[i] = input[i] + attn_out[i];
    }

    /*LN2*/
    layer_norm_opencl(residual, ln2_out, ln2_w, ln2_b);

    /*MLP*/
    mlp_block(ln2_out, mlp_out, mlp1_w, mlp1_b, mlp2_w, mlp2_b);

    /*Residual2*/
    for (int i = 0; i < tokens * embed_dim; i++) {
        output[i] = residual[i] + mlp_out[i];
    }

    free(ln1_out); free(attn_out); free(residual); free(ln2_out); free(mlp_out);
}

////////////////////////////////////// Model Architecture //////////////////////////////////////
void ViT_opencl(ImageData *image, Network *networks, float **probabilities) {
    time_t start, end;
    int token_size = ((img_size / patch_size) * (img_size / patch_size) + 1);
    float *layer[4];
    float *enc_layer[12];
    float *enc_output;
    int  hidden_dim = ((int)(embed_dim * mlp_ratio));
    //printf("%d %d = %d\n", token_size, hidden_dim, token_size * hidden_dim);

    for (int i = 0; i < 4; i++) {
        layer[i] = (float *)malloc(sizeof(float) * size[i]);
    }
    for (int i = 0; i < 12; i++) {
        enc_layer[i] = (float *)malloc(sizeof(float) * enc_size);
    }
    enc_output = (float *)malloc(sizeof(float) * enc_size);

    for (int i = 0; i < image->n; i++) {
        /*patch embedding*/
        Conv2d_opencl(image[i].data, layer[0], networks[1], networks[2]);
        /*flatten and transpose*/
        flatten_transpose(layer[0], layer[1]);
        /*prepend class token*/
        class_token(layer[1], layer[2], networks[0]);
        /*position embedding*/
        pos_emb(layer[2], layer[3], networks[3]);

        /*Encoder - 12 Layers*/
        Encoder_opencl(layer[3], enc_layer[0],
            networks[4], networks[5], networks[6], networks[7],
            networks[8], networks[9], networks[10], networks[11],
            networks[12], networks[13], networks[14], networks[15]);

        Encoder_opencl(enc_layer[0], enc_layer[1],
            networks[16], networks[17], networks[18], networks[19],
            networks[20], networks[21], networks[22], networks[23],
            networks[24], networks[25], networks[26], networks[27]);

        Encoder_opencl(enc_layer[1], enc_layer[2],
            networks[28], networks[29], networks[30], networks[31],
            networks[32], networks[33], networks[34], networks[35],
            networks[36], networks[37], networks[38], networks[39]);

        Encoder_opencl(enc_layer[2], enc_layer[3],
            networks[40], networks[41], networks[42], networks[43],
            networks[44], networks[45], networks[46], networks[47],
            networks[48], networks[49], networks[50], networks[51]);

        Encoder_opencl(enc_layer[3], enc_layer[4],
            networks[52], networks[53], networks[54], networks[55],
            networks[56], networks[57], networks[58], networks[59],
            networks[60], networks[61], networks[62], networks[63]);

        Encoder_opencl(enc_layer[4], enc_layer[5],
            networks[64], networks[65], networks[66], networks[67],
            networks[68], networks[69], networks[70], networks[71],
            networks[72], networks[73], networks[74], networks[75]);

        Encoder_opencl(enc_layer[5], enc_layer[6],
            networks[76], networks[77], networks[78], networks[79],
            networks[80], networks[81], networks[82], networks[83],
            networks[84], networks[85], networks[86], networks[87]);

        Encoder_opencl(enc_layer[6], enc_layer[7],
            networks[88], networks[89], networks[90], networks[91],
            networks[92], networks[93], networks[94], networks[95],
            networks[96], networks[97], networks[98], networks[99]);

        Encoder_opencl(enc_layer[7], enc_layer[8],
            networks[100], networks[101], networks[102], networks[103],
            networks[104], networks[105], networks[106], networks[107],
            networks[108], networks[109], networks[110], networks[111]);

        Encoder_opencl(enc_layer[8], enc_layer[9],
            networks[112], networks[113], networks[114], networks[115],
            networks[116], networks[117], networks[118], networks[119],
            networks[120], networks[121], networks[122], networks[123]);

        Encoder_opencl(enc_layer[9], enc_layer[10],
            networks[124], networks[125], networks[126], networks[127],
            networks[128], networks[129], networks[130], networks[131],
            networks[132], networks[133], networks[134], networks[135]);

        Encoder_opencl(enc_layer[10], enc_layer[11],
            networks[136], networks[137], networks[138], networks[139],
            networks[140], networks[141], networks[142], networks[143],
            networks[144], networks[145], networks[146], networks[147]);

        layer_norm_opencl(enc_layer[11], enc_output, networks[148], networks[149]);
        /* Token 값 추출 */
        float *cls_token = (float *)malloc(sizeof(float) * embed_dim);
        float *cls_output = (float *)malloc(sizeof(float) * num_classes);
        memcpy(cls_token, enc_output, sizeof(float) * embed_dim);

        linear_layer(cls_token, cls_output, 1, embed_dim, num_classes, networks[150], networks[151]);
        /* 확률분포 추출 */
        Softmax(cls_output, probabilities[i], num_classes);
    }
}
