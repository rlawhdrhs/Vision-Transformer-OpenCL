#pragma warning(disable : 4996)
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <CL/cl.h>
#include "Network.h"

#define img_size 224
#define patch_size 16
#define embed_dim 768
#define eps 1e-6
#define NUM_CHUNK 3

#define CHECK_ERROR(err) \
    if (err != CL_SUCCESS) { \
        printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
        exit(EXIT_FAILURE); \
    }

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

cl_event enqueue_reduction_stage(
    cl_command_queue queue,
    cl_kernel kernel,
    cl_mem input_buf,          // Arg 0 (g_num)
    cl_mem output_buf,         // Arg 1 (g_sum)
    size_t local_size,         // Local Work Size (for Arg 2)
    int total_num,             // Arg 3 (TotalNum)
    size_t global_size,        // Global Work Size
    const size_t *global_offset, 
    int num_events_in_wait_list,
    const cl_event *event_wait_list,
    cl_event *event_out) 
{
    cl_int err;
    size_t local_mem_size = local_size * sizeof(double);

    // Arg 0 (g_num)
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buf);
    // Arg 1 (g_sum)
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buf);
    // Arg 2 (l_sum, Local Memory)
    err = clSetKernelArg(kernel, 2, local_mem_size, NULL);
    // Arg 3 (TotalNum)
    err = clSetKernelArg(kernel, 3, sizeof(int), &total_num);

    // 커널 실행 큐에 추가
    err = clEnqueueNDRangeKernel(queue, kernel, 1, global_offset, &global_size, &local_size,
        num_events_in_wait_list, event_wait_list, event_out);
    return *event_out;
}

void layer_norm_opencl(float *input, float *output, Network weight, Network bias) 
{
    int token = ((img_size / patch_size) * (img_size / patch_size)) + 1;

    cl_int err;

    // Platform ID
    cl_platform_id platform;
    err = clGetPlatformIDs(1, &platform, NULL);
    CHECK_ERROR(err);

    // Device ID
    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    CHECK_ERROR(err);

    // Create Context
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_ERROR(err);

    // Create Command Queue
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, 0, &err);
    CHECK_ERROR(err);

    // Create Program Object
    size_t kernel_source_size;
    char *kernel_source = get_source_code("norm_kernel.cl", &kernel_source_size);
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&kernel_source, &kernel_source_size, &err);
    CHECK_ERROR(err);

    // Build Program
    err = clBuildProgram(program, 1, &device, "", NULL, NULL);
    build_error(program, device, err);
    CHECK_ERROR(err);

    cl_kernel kernel = clCreateKernel(program, "norm_reduction", &err);
    CHECK_ERROR(err);

    int work_size = embed_dim / NUM_CHUNK;
    const size_t local_mem_size = work_size * sizeof(float);

    const int N_CHUNK = embed_dim / NUM_CHUNK;
    const int GROUP_NUM = N_CHUNK / work_size;

    // --- 버퍼 및 이벤트 관리 ---
    cl_mem final_buffers[NUM_CHUNK];
    cl_event chunk_events[NUM_CHUNK];

    cl_mem input_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * token * embed_dim, input, &err);
    CHECK_ERROR(err);

    // LayerNorm 가중치와 편향 버퍼 생성
    cl_mem weight_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * embed_dim, weight.data, &err);
    CHECK_ERROR(err);
    cl_mem bias_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * embed_dim, bias.data, &err);
    CHECK_ERROR(err);


    // 평균과 분산을 저장할 버퍼
    cl_mem mean_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * token, NULL, &err);
    CHECK_ERROR(err);
    cl_mem variance_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * token, NULL, &err);
    CHECK_ERROR(err);

    // 필요한 커널 추가:
    cl_kernel layernorm_kernel = clCreateKernel(program, "layernorm_apply", &err);
    CHECK_ERROR(err);

    // Global Work Size: 토큰의 개수 (각 토큰이 하나의 Work Group)
    const size_t global_size[1] = { (size_t)token * NUM_CHUNK }; // 또는 embed_dim을 나누는 Work Group 수
    const size_t local_size[1] = { (size_t)work_size }; // 각 Work Group의 크기

    // 만약 Single-Kernel (Row-Wise) 방식을 사용한다면:
    // 1. 커널 인자 설정
    err = clSetKernelArg(layernorm_kernel, 0, sizeof(cl_mem), &input_buf);
    // ... 나머지 인자들 (output, weight_buf, bias_buf, embed_dim, epsilon 등) 설정

    // 2. 커널 실행 (각 토큰을 병렬로 처리)
    // GWS: 토큰 수, LWS: embed_dim을 나눌 수 있는 수
    const size_t single_kernel_gws[1] = { (size_t)token * embed_dim };
    const size_t single_kernel_lws[1] = { 256 }; // 예시

    err = clEnqueueNDRangeKernel(queue, layernorm_kernel, 1, NULL, single_kernel_gws, single_kernel_lws, 0, NULL, NULL);
    CHECK_ERROR(err);

    // 3. 결과 읽기
    err = clEnqueueReadBuffer(queue, output_buf, CL_TRUE, 0, sizeof(float) * token * embed_dim, output, 0, NULL, NULL);
    CHECK_ERROR(err);


    
    /*for (int i = 0; i < token; ++i)
    {
        cl_event events[NUM_CHUNK];
        for (int j = 0; j < NUM_CHUNK; ++j)
        {
            const size_t global_offset = (size_t)j * N_CHUNK;

            cl_mem intermediate_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * GROUP_NUM, NULL, &err);
            final_buffers[j] = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float), NULL, &err);

            if (j == 0)
                enqueue_reduction_stage(queue, kernel,
                    input_buf, intermediate_buf,
                    work_size, N_CHUNK,
                    N_CHUNK, &global_offset,
                    0, NULL, &events[j]);
            else if (j == NUM_CHUNK - 1)
                enqueue_reduction_stage(queue, kernel,
                    input_buf, intermediate_buf,
                    work_size, N_CHUNK,
                    N_CHUNK, &global_offset,
                    0, &events[j - 1], &chunk_events[i]);
            else
                enqueue_reduction_stage(queue, kernel,
                    input_buf, intermediate_buf,
                    work_size, N_CHUNK,
                    N_CHUNK, &global_offset,
                    0, &events[j - 1], &events[j]);
        }

    }*/


    clReleaseMemObject(input_buf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(kernel_source);


    free(output);
}