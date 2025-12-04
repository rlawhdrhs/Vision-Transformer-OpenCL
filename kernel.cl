#define EMBED_DIM 768
#define EPSILON 1e-5f
#define TS 16
#define SOFTMAX_WG_SIZE 256

__kernel void layer_norm_kernel(
    __global float *g_input,
    __global float *g_output,
    __global float *g_weight,
    __global float *g_bias,
    __local float *l_sum,
    const int TokenSize,
    const int GlobalOffset
)
{
    int local_id = get_local_id(0);
    int global_id = get_global_id(0);
    int token_idx = get_group_id(0);
    int local_size = get_local_size(0);

    float sum_val = 0.0f;       // E(x)
    float sum_sq_val = 0.0f;    // E(x^2)

    for (int i = local_id; i < EMBED_DIM; i += local_size) {
        float val = g_input[token_idx * EMBED_DIM + i];
        sum_val += val;
        sum_sq_val += val * val;
    }

    l_sum[local_id] = sum_val;
    l_sum[local_id + local_size] = sum_sq_val;

    barrier(CLK_LOCAL_MEM_FENCE);


    // Reduction

    for (int p = local_size / 2; p >= 1; p = p >> 1) {
        if (local_id < p) {

            // E(x)
            l_sum[local_id] += l_sum[local_id + p];

            // E(x^2)
            l_sum[local_id + local_size] += l_sum[local_id + local_size + p];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }


    float mean = 0.0f;
    float inv_std = 0.0f;

    if (local_id == 0) {
        float final_sum = l_sum[0];
        float final_sum_sq = l_sum[local_size];

        // E(x)
        mean = final_sum / (float)EMBED_DIM;

        // Var(x) = E(x^2) - (E(x))^2
        float var = final_sum_sq / (float)EMBED_DIM - mean * mean;
        inv_std = 1.0f / sqrt(var);

        l_sum[0] = mean;
        l_sum[1] = inv_std;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    mean = l_sum[0];
    inv_std = l_sum[1];


    for (int i = local_id; i < EMBED_DIM; i += local_size) {
        int idx = token_idx * EMBED_DIM + i;
        float normalized_val = (g_input[idx] - mean) * inv_std;

        g_output[idx] = normalized_val * g_weight[i] + g_bias[i];
    }
}

__kernel void gemm(__global float *A,
	__global float *B,
	__global float *C,
	const int M, // ROW_A
	const int N, // COL_A, ROW_B
	const int K) // COL_B
{
	
	// C[M x K]의 좌표 (ci: 열 인덱스, cj: 행 인덱스)
	const int i = get_local_id(0);
	const int j = get_local_id(1);

	const int gi = get_group_id(0) * TS + i;    // TS는 tile 크기, 2
	const int gj = get_group_id(1) * TS + j;

	__local float Asub[TS][TS];
	__local float Bsub[TS][TS];

	float sum = 0.0f;
	
    // 타일 순회
	for (int t = 0; t <N; t += TS) {
		const int ti = t + i;    // for loop 내에서 타일 내 좌표를 확인
		const int tj = t + j;

		Asub[j][i] = A[N * gj + ti];
		Bsub[j][i] = B[K * tj + gi];

		barrier(CLK_LOCAL_MEM_FENCE);

		for (int k=0; k < TS; k++)
			sum += Asub[j][k] * Bsub[k][i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}
	C[K * gj + gi] = sum;
}

__kernel void Conv2d_Kernel(__global float* input,
                                       __global float* output,
                                       __global float* weight,
                                       __global float* bias,
                                       int img_size, int patch_size, int in_chans, int embed_dim,
                                       int output_size) { // output_size는 img_size / patch_size와 동일

    // Work-Item ID로 출력 좌표 계산
    int oc = get_global_id(0); // Output Channel (Embed Dim)
    int oh = get_global_id(1); // Output Height (Patch Row)
    int ow = get_global_id(2); // Output Width (Patch Col)

    if (oc >= embed_dim || oh >= output_size || ow >= output_size)
        return;

    // 1. Bias 초기화
    float sum = bias[oc];

    // 상수 미리 계산
    const int input_channel_size = img_size * img_size; // H * W
    const int output_patch_size_sq = output_size * output_size; // H_out * W_out
    
    // Weight 인덱싱 관련 상수
    const int weight_oc_offset = oc * in_chans * patch_size * patch_size;
    const int kernel_patch_size_sq = patch_size * patch_size; // K_H * K_W

    // 2. Convolution 연산 (Work-Item 내에서 순차 처리)
    for (int ic = 0; ic < in_chans; ++ic) {
        
        // 입력 및 가중치 오프셋 계산
        const int input_ic_offset = ic * input_channel_size;
        const int weight_ic_offset = weight_oc_offset + (ic * kernel_patch_size_sq);
        
        for (int kh = 0; kh < patch_size; ++kh) {
            
            int ih = oh * patch_size + kh;
            int input_ih_offset = input_ic_offset + (ih * img_size); // C + H 오프셋

            // Weight의 Kh 오프셋
            const int weight_kh_offset = weight_ic_offset + (kh * patch_size);

            for (int kw = 0; kw < patch_size; ++kw) {
                int iw = ow * patch_size + kw;

                int input_idx = input_ih_offset + iw;
                int kernel_idx = weight_kh_offset + kw;

                sum += input[input_idx] * weight[kernel_idx];
            }
        }
    }

    // 3. 결과 저장
    int output_idx = (oc * output_patch_size_sq) + (oh * output_size) + ow;
    output[output_idx] = sum;
}

__kernel void FlattenTranspose_Kernel(__global float* input,
                                      __global float* output,
                                      int output_size, int embed_dim) {

    // Global ID로 출력 인덱스 계산: [patch_idx, oc]
    // get_global_id(0)을 patch_idx로, get_global_id(1)을 oc로 사용합니다.
    int patch_idx = get_global_id(0); // 0 ~ num_patches - 1 (196)
    int oc = get_global_id(1);        // 0 ~ embed_dim - 1 (767)

    int num_patches = output_size * output_size;

    if (patch_idx >= num_patches || oc >= embed_dim)
        return;

    // 출력 인덱스 (Patch, Embed)에 해당하는 입력 인덱스 (C, H, W)를 역산합니다.
    
    // 1. patch_idx를 (oh, ow)로 역변환
    int oh = patch_idx / output_size;
    int ow = patch_idx % output_size;

    // 2. 입력 (C, H, W) 인덱스 계산
    // 기존 입력: (oc, oh, ow)
    int idx_input = (oc * output_size + oh) * output_size + ow;
    
    // 3. 출력 (Patch, Embed) 인덱스 계산
    // 원하는 출력: (patch_idx, oc)
    int idx_output = patch_idx * embed_dim + oc;

    output[idx_output] = input[idx_input];
}

__kernel void MHA_gemm_kernel(
    __global float *A, __global float *B, __global float *C,
    const int M, const int N, const int K, 
    const int A_row_stride, const int B_row_stride, const int C_row_stride, 
    const int A_offset, const int B_offset, const int C_offset, 
    const int transpose_B, const float scale_factor 
)
{
    // Local ID
    const int i = get_local_id(0); 
    const int j = get_local_id(1); 
    
    // Global Output Index
    const int gi = get_group_id(0) * TS + i; // Col (K 방향)
    const int gj = get_group_id(1) * TS + j; // Row (M 방향)

    // 로컬 메모리 (반드시 매크로 상수 TS 사용)
    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];

    float sum = 0.0f;
    
    // 타일 순회 (N축)
    // 주의: N이 TS의 배수가 아닐 수 있으므로 올림 처리된 횟수만큼 반복
    int num_tiles = (N + TS - 1) / TS;

    for (int t = 0; t < num_tiles; t++) {
        const int tiled_n_idx = t * TS;
        const int ti = tiled_n_idx + i; // A의 열
        const int tj = tiled_n_idx + j; // B의 행

        // A 로딩 (Row: gj, Col: ti)
        // 범위 체크를 여기서 수행하여 패딩 영역은 0으로 채움
        if (gj < M && ti < N) {
            int A_idx = A_offset + gj * A_row_stride + ti;
            Asub[j][i] = A[A_idx];
        } else {
            Asub[j][i] = 0.0f;
        }

        // B 로딩
        if (transpose_B) { 
            // B^T (Row: tj, Col: gi) -> 실제 B 메모리 (Row: gi, Col: tj)
            if (gi < K && tj < N) {
                int B_idx = gi * B_row_stride + B_offset + tj;
                Bsub[j][i] = B[B_idx];
            } else {
                Bsub[j][i] = 0.0f;
            }
        } else { 
            // B (Row: tj, Col: gi)
            if (tj < N && gi < K) {
                int B_idx = tj * B_row_stride + B_offset + gi;
                Bsub[j][i] = B[B_idx];
            } else {
                Bsub[j][i] = 0.0f;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // 연산 (Local Memory 내)
        // TS 크기만큼 모두 수행 (0으로 패딩되었으므로 안전)
        #pragma unroll
        for (int k = 0; k < TS; k++) {
            sum += Asub[j][k] * Bsub[k][i];    
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // 최종 결과 저장
    if (gj < M && gi < K) {
        float final_sum = sum * scale_factor;
        C[C_offset + gj * C_row_stride + gi] = final_sum;
    }
}

// Softmax: 한 행(Row)을 하나의 WorkGroup(256 threads)이 처리한다고 가정
#define SOFTMAX_WG_SIZE 256

__kernel void softmax_reduction_kernel(
    __global float* scores, 
    int tokens,
    int score_offset)
{
    // global_id(0)가 행(row) 인덱스
    // get_local_id(0)가 컬럼 처리를 위한 스레드 ID
    // 2D Range로 실행해야 함: Global[rows, 256], Local[1, 256] 
    // 하지만 호스트 코드를 1D로 짜셨으니, 아래와 같이 1D 형태에 맞춰 수정합니다.
    
    // 수정된 호스트 로직 가정: 
    // GlobalSize = tokens * 256
    // LocalSize = 256
    // GroupID = 처리할 행(Row) 인덱스 (Token Index)
    
    int row_idx = get_group_id(0);
    int lid = get_local_id(0);
    
    if (row_idx >= tokens) return; // 행 자체가 범위를 벗어나면 종료 (배리어 영향 없음)

    int row_offset = score_offset + row_idx * tokens;
    
    __local float shared_data[SOFTMAX_WG_SIZE];

    // ----------------------------------------------------
    // 1. Max Reduction
    // ----------------------------------------------------
    float max_val = -INFINITY;

    // tokens가 256보다 클 경우를 대비해 loop 처리
    for (int i = lid; i < tokens; i += SOFTMAX_WG_SIZE) {
        float val = scores[row_offset + i];
        max_val = fmax(max_val, val);
    }
    shared_data[lid] = max_val;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Tree Reduction (256 -> 1)
    for (int s = SOFTMAX_WG_SIZE / 2; s > 0; s >>= 1) {
        if (lid < s) {
            shared_data[lid] = fmax(shared_data[lid], shared_data[lid + s]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    max_val = shared_data[0];
    barrier(CLK_LOCAL_MEM_FENCE); // 모든 스레드가 max_val을 알 때까지 대기

    // ----------------------------------------------------
    // 2. Exp & Sum Reduction
    // ----------------------------------------------------
    float sum_val = 0.0f;
    for (int i = lid; i < tokens; i += SOFTMAX_WG_SIZE) {
        float val = exp(scores[row_offset + i] - max_val);
        scores[row_offset + i] = val; // Write back exp value
        sum_val += val;
    }
    shared_data[lid] = sum_val;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Tree Reduction (Sum)
    for (int s = SOFTMAX_WG_SIZE / 2; s > 0; s >>= 1) {
        if (lid < s) {
            shared_data[lid] += shared_data[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    sum_val = shared_data[0];
    // barrier 불필요 (sum_val은 이제 shared_data[0]에서 읽기만 함)

    // ----------------------------------------------------
    // 3. Normalization
    // ----------------------------------------------------
    float inv_sum = 1.0f / sum_val;
    for (int i = lid; i < tokens; i += SOFTMAX_WG_SIZE) {
        scores[row_offset + i] *= inv_sum;
    }
}