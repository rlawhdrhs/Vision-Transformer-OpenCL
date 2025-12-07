#define TS 32

// 1. Bias Broadcast
__kernel void add_bias_broadcast_kernel(
    __global float* data,
    __global float* bias,
    int rows,        // Batch * Tokens
    int cols,        // Embed
    int bias_offset  // Q=0, K=Embed, V=2*Embed
) {
    int row = get_global_id(0);
    int col = get_global_id(1);

    if (row >= rows || col >= cols) return;

    int data_idx = row * cols + col;
    int bias_idx = bias_offset + col;

    data[data_idx] += bias[bias_idx];
}

// 2. Layer Norm
__kernel void layer_norm_kernel(
    __global float *g_input, __global float *g_output,
    __global float *g_weight, __global float *g_bias,
    __local float *l_sum,
    const int TokenSize, const int EmbedDim
) {
    int local_id = get_local_id(0);
    int group_id = get_group_id(0); // Token Index
    int local_size = get_local_size(0);
    int offset = group_id * EmbedDim;

    float sum_val = 0.0f;
    float sum_sq_val = 0.0f;

    for (int i = local_id; i < EmbedDim; i += local_size) {
        float val = g_input[offset + i];
        sum_val += val;
        sum_sq_val += val * val;
    }
    l_sum[local_id] = sum_val;
    l_sum[local_id + local_size] = sum_sq_val;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int p = local_size / 2; p >= 1; p >>= 1) {
        if (local_id < p) {
            l_sum[local_id] += l_sum[local_id + p];
            l_sum[local_id + local_size] += l_sum[local_id + local_size + p];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float mean = l_sum[0] / EmbedDim;
    float var = l_sum[local_size] / EmbedDim - mean * mean;
    float inv_std = 1.0f / sqrt(var + 1e-6f);

    for (int i = local_id; i < EmbedDim; i += local_size) {
        g_output[offset + i] = (g_input[offset + i] - mean) * inv_std * g_weight[i] + g_bias[i];
    }
}

// 3. Conv2d Batched
__kernel void Conv2d_Batched_Kernel(
    __global float* input, __global float* output,
    __global float* weight, __global float* bias,
    int img_size, int patch_size, int in_chans, int embed_dim, int output_size) 
{
    int oc = get_global_id(0);
    int oh = get_global_id(1);
    int global_z = get_global_id(2);
    int batch_idx = global_z / output_size;
    int ow = global_z % output_size;

    if (oc >= embed_dim || oh >= output_size) return;

    int in_batch_off = batch_idx * in_chans * img_size * img_size;
    int out_batch_off = batch_idx * embed_dim * output_size * output_size;
    
    float sum = bias[oc];
    int w_oc_off = oc * in_chans * patch_size * patch_size;

    for (int ic = 0; ic < in_chans; ++ic) {
        int in_ic_off = in_batch_off + ic * img_size * img_size;
        int w_ic_off = w_oc_off + ic * patch_size * patch_size;
        for (int kh = 0; kh < patch_size; ++kh) {
            int in_ih_off = in_ic_off + (oh * patch_size + kh) * img_size;
            int w_kh_off = w_ic_off + kh * patch_size;
            for (int kw = 0; kw < patch_size; ++kw) {
                sum += input[in_ih_off + ow * patch_size + kw] * weight[w_kh_off + kw];
            }
        }
    }
    output[out_batch_off + (oc * output_size + oh) * output_size + ow] = sum;
}

// 4. Flatten Batched
__kernel void FlattenTranspose_Batched_Kernel(
    __global float* input, __global float* output,
    int output_size, int embed_dim, int num_patches) 
{
    int global_x = get_global_id(0);
    int oc = get_global_id(1);
    int batch_idx = global_x / num_patches;
    int patch_idx = global_x % num_patches;

    if (oc >= embed_dim) return;

    int oh = patch_idx / output_size;
    int ow = patch_idx % output_size;
    
    // NCHW Read
    int in_idx = batch_idx * (embed_dim * output_size * output_size) + (oc * output_size + oh) * output_size + ow;
    // NLC Write
    int out_idx = batch_idx * (num_patches * embed_dim) + patch_idx * embed_dim + oc;

    output[out_idx] = input[in_idx];
}

// 5. GEMM (C = A * B^T or A * B)
__kernel void MHA_gemm_kernel(
    __global float *A, __global float *B, __global float *C,
    const int M, const int N, const int K, 
    const int A_stride, const int B_stride, const int C_stride, 
    const int A_off, const int B_off, const int C_off, 
    const int transB, const float scale)
{
    int i = get_local_id(0);
    int j = get_local_id(1);
    int gi = get_group_id(0) * TS + i; // Col of C
    int gj = get_group_id(1) * TS + j; // Row of C

    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];
    
    float sum = 0.0f;
    int num_tiles = (N + TS - 1) / TS;

    for (int t = 0; t < num_tiles; t++) {
        int tiled_n = t * TS;
        int ti = tiled_n + i;
        int tj = tiled_n + j;

        // Load A [Row, Inner]
        if (gj < M && ti < N) Asub[j][i] = A[A_off + gj * A_stride + ti];
        else Asub[j][i] = 0.0f;

        // Load B [Col, Inner] (if TransB)
        if (transB) {
            // B is [K, N] logically for us (Weight[Out, In])
            // We want B^T. Kernel expects B to be stored such that we read Row gi, Col tj?
            // Correct mapping: B[gi * B_stride + tj] accesses Weight[OutFeat, InnerDim]
            if (gi < K && tj < N) Bsub[j][i] = B[B_off + gi * B_stride + tj];
            else Bsub[j][i] = 0.0f;
        } else {
            // Standard Mul: B[Inner, Col]
            if (tj < N && gi < K) Bsub[j][i] = B[B_off + tj * B_stride + gi];
            else Bsub[j][i] = 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TS; k++) sum += Asub[j][k] * Bsub[k][i];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (gj < M && gi < K) C[C_off + gj * C_stride + gi] = sum * scale;
}

__kernel void mha_score_kernel(
    __global float* Q, __global float* K, __global float* scores,
    int tokens, int head_dim, int num_heads, float scale)
{
    int bh = get_global_id(0); // Batch * Heads
    int i  = get_global_id(1); // Token Q index (Row)

    if (i >= tokens) return;

    // 현재 배치와 헤드 계산
    int batch_idx = bh / num_heads;
    int head_idx  = bh % num_heads;

    // 메모리 오프셋 계산
    // Q, K 구조: [Batch, Tokens, EmbedDim] (EmbedDim 안에 Heads가 인터리빙 되어 있음)
    // 순차 코드: Q[t * embed_dim + h * head_dim + d]
    
    int embed_dim = num_heads * head_dim;
    int batch_offset = batch_idx * tokens * embed_dim;
    int head_offset = head_idx * head_dim;

    // Score Output: [Batch, Heads, Tokens, Tokens]
    // Score Offset: batch * (Heads*Tokens*Tokens) + head * (Tokens*Tokens) + i * Tokens
    int score_row_start = (batch_idx * num_heads * tokens * tokens) + 
                          (head_idx * tokens * tokens) + 
                          (i * tokens);

    // 모든 K 토큰에 대해 내적 수행 (Loop j)
    for (int j = 0; j < tokens; ++j) {
        float sum = 0.0f;
        int q_ptr = batch_offset + i * embed_dim + head_offset;
        int k_ptr = batch_offset + j * embed_dim + head_offset;

        for (int d = 0; d < head_dim; ++d) {
            sum += Q[q_ptr + d] * K[k_ptr + d];
        }
        scores[score_row_start + j] = sum * scale;
    }
}

__kernel void mha_softmax_kernel(__global float* scores, int tokens) {
    int bh = get_global_id(0);
    int i = get_global_id(1); // Row index
    
    if (i >= tokens) return;

    int row_idx = bh * tokens * tokens + i * tokens;

    // 1. Max 찾기
    float max_val = -1e30f;
    for (int j = 0; j < tokens; j++) {
        max_val = fmax(max_val, scores[row_idx + j]);
    }

    // 2. Exp & Sum
    float sum_exp = 0.0f;
    for (int j = 0; j < tokens; j++) {
        float val = exp(scores[row_idx + j] - max_val);
        scores[row_idx + j] = val;
        sum_exp += val;
    }

    // 3. Normalize
    float inv_sum = 1.0f / (sum_exp + 1e-6f);
    for (int j = 0; j < tokens; j++) {
        scores[row_idx + j] *= inv_sum;
    }
}

__kernel void mha_context_kernel(
    __global float* scores, __global float* V, __global float* output,
    int tokens, int head_dim, int num_heads)
{
    int b = get_global_id(0);
    int i = get_global_id(1); // Output Token Index
    int e = get_global_id(2); // Embed Dim Index (0..767)

    if (i >= tokens || e >= (num_heads * head_dim)) return;

    int embed_dim = num_heads * head_dim;
    
    // 현재 e가 속한 헤드(h)와 헤드 내부 차원(d) 계산
    int h = e / head_dim;
    int d = e % head_dim;

    // Score 읽기 위치: [Batch, Head, i, :]
    int score_base = (b * num_heads * tokens * tokens) + (h * tokens * tokens) + (i * tokens);
    
    // V 읽기 위치: [Batch, :, Embed] -> V[b, j, e]가 아니라 Head별로 맞춰야 함.
    // 순차 코드 V 구조: [Batch, Tokens, Embed]. V[t * embed + h*head_dim + d]
    // 우리가 필요한 V의 값은 특정 Head h, 차원 d에 대해 모든 토큰 j를 순회.
    int v_batch_offset = b * tokens * embed_dim;
    int v_head_offset = h * head_dim; 

    float sum = 0.0f;
    for (int j = 0; j < tokens; ++j) {
        float s = scores[score_base + j];
        float v_val = V[v_batch_offset + j * embed_dim + v_head_offset + d];
        sum += s * v_val;
    }

    output[b * tokens * embed_dim + i * embed_dim + e] = sum;
}

// 6. Residual Add (Elementwise)
__kernel void add_residual_kernel(__global float* in, __global float* out, int total) {
    int idx = get_global_id(0);
    if (idx < total) out[idx] += in[idx];
}

// 7. Softmax
__kernel void softmax_reduction_kernel(__global float* scores, int tokens, int offset) {
    int row_idx = get_group_id(0);
    int lid = get_local_id(0);
    if (row_idx >= tokens) return;

    int row_off = offset + row_idx * tokens;
    __local float shared[256];

    // Max
    float max_val = -1e30f; // -INFINITY safely
    for (int i = lid; i < tokens; i += 256) max_val = fmax(max_val, scores[row_off + i]);
    shared[lid] = max_val;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = 128; s > 0; s >>= 1) {
        if (lid < s) shared[lid] = fmax(shared[lid], shared[lid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    max_val = shared[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Exp Sum
    float sum_val = 0.0f;
    for (int i = lid; i < tokens; i += 256) {
        float val = exp(scores[row_off + i] - max_val);
        scores[row_off + i] = val;
        sum_val += val;
    }
    shared[lid] = sum_val;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = 128; s > 0; s >>= 1) {
        if (lid < s) shared[lid] += shared[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Normalize
    float inv_sum = 1.0f / (shared[0] + 1e-6f);
    for (int i = lid; i < tokens; i += 256) scores[row_off + i] *= inv_sum;
}

// 8. Linear (Dense)
__kernel void linear_tiled_float4(
    int M, int N, int K,
    __global const float* A,
    __global const float* B,
    __global const float* bias,
    __global float* C,
    int weight_offset) 
{

    const int row = get_local_id(1); 
    const int col = get_local_id(0); 
    
    // Global Index
    const int globalRow = TS * get_group_id(1) + row; 
    const int globalCol = TS * get_group_id(0) + (col * 4); // float4 기준이므로 *4

    // Local Memory (Padding +1 to avoid bank conflicts)
    __local float Asub[TS][TS + 1];
    __local float Bsub[TS][TS + 1];

    float4 acc = (float4)(0.0f);

    const int numTiles = (K + TS - 1) / TS;

    for (int t = 0; t < numTiles; t++) {
        const int tiledCol = t * TS + col * 4;
        
        #pragma unroll
        for (int i = 0; i < 4; i++) {
             int c = tiledCol + i;
             if (globalRow < M && c < K)
                 Asub[row][col * 4 + i] = A[globalRow * K + c];
             else
                 Asub[row][col * 4 + i] = 0.0f;
        }
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int w_row = globalCol + i + weight_offset; // Output Ch
            int w_col = t * TS + row;                  // Input Ch (K)

            if (w_row < (N + weight_offset) && w_col < K)
                Bsub[col * 4 + i][row] = B[w_row * K + w_col];
            else
                Bsub[col * 4 + i][row] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TS; k++) {
            float valA = Asub[row][k];
            float4 valB;
            valB.x = Bsub[col * 4 + 0][k];
            valB.y = Bsub[col * 4 + 1][k];
            valB.z = Bsub[col * 4 + 2][k];
            valB.w = Bsub[col * 4 + 3][k];

            acc += valA * valB;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (globalRow < M && globalCol < N) {
        float4 b_val;
        b_val.x = bias[globalCol + 0 + weight_offset];
        b_val.y = bias[globalCol + 1 + weight_offset];
        b_val.z = bias[globalCol + 2 + weight_offset];
        b_val.w = bias[globalCol + 3 + weight_offset];
        
        __global float4* C_ptr = (__global float4*)&C[globalRow * N + globalCol];
        *C_ptr = acc + b_val;
    }
}

#define TS 32
#define WPT 4
#define RTS (TS / WPT) // 8

__kernel void linear_register_blocked_4x4(
    int M, int N, int K,
    __global const float* A,
    __global const float* B,
    __global const float* bias,
    __global float* C,
    int weight_offset)
{
    // Thread IDs
    const int row = get_local_id(1); // 0..7
    const int col = get_local_id(0); // 0..7
    const int globalRow = (get_group_id(1) * TS) + row; // Row start for this thread
    const int globalCol = (get_group_id(0) * TS) + col; // Col start for this thread

    // Local Memory (Shared Memory)
    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];

    // Registers (Accumulators for 4x4 output)
    float acc[WPT][WPT];
    for (int w = 0; w < WPT; w++) {
        for (int r = 0; r < WPT; r++) {
            acc[w][r] = 0.0f;
        }
    }

    // Loop over all tiles
    const int numTiles = K / TS;
    for (int t = 0; t < numTiles; t++) {
        
        // ---------------------------------------------------------
        // 1. Cooperative Loading (Global -> Local)
        // ---------------------------------------------------------
        // 64 threads need to load 32x32 (1024) elements.
        // Each thread loads 1024 / 64 = 16 elements.
        // Unrolled for performance.
        
        const int tiledK = t * TS;

        #pragma unroll
        for (int l = 0; l < WPT; l++) {
             // Load A: [TS, TS] tile
             int r_A = row + l * RTS; // row + 0, row + 8, row + 16, row + 24
             int c_A = col;           // Stride access might be bad here but keeps logic simple
             // Optimized loading pattern can be used, but let's stick to correctness first
             // Ideally we load float4, but standard load loop is safer for dimensions
             
             // Actual Load A
             // We need to cover [row*RTS .. row*RTS + WPT] ? No.
             // We use (row, col) as ID to cover the whole TSxTS block
             
             // Simple Tiled Loading:
             // Thread (row, col) loads Asub[row][col] ... No, 64 threads vs 1024 elements.
             // We use (row*8 + col) as linear ID
             int tid = row * RTS + col;
             int r_tile = tid / TS; // 0..31
             int c_tile = tid % TS; // 0..31
             
             // But we need each thread to load 16 elements?
             // Let's use the explicit loop defined by WPT
             
             int tiled_r = row + l * RTS;
             int tiled_c = col; // Strided load for coalescing? Let's load 4x float4
             
             // Let's map 8x8 threads to load 32x32 tile
             // Each thread loads 4 float4s?
             // A: [M, K], B: [N, K]
        }
        
        // --- Simplified Loading Strategy (Standard) ---
        // Each thread loads 4 floats for A and 4 floats for B
        #pragma unroll
        for (int w = 0; w < WPT; w++) {
            // Load A [32, 32]
            // We want A[globalRow + w*RTS][tiledK + col] ? No.
            
            // Just use linear tiling for loading
            int tid = row * RTS + col; // 0..63
            // We need to load 1024 elements. 64 threads. 16 elements per thread.
            // 16 elements = 4 x float4.
            
            int id_per_thread = w * 64 + tid;
            int r_idx = id_per_thread / TS;
            int c_idx = id_per_thread % TS;
            
            // Load A
            int g_r = (get_group_id(1) * TS) + r_idx;
            int g_c = tiledK + c_idx;
            if(g_r < M && g_c < K) Asub[r_idx][c_idx] = A[g_r * K + g_c];
            else                   Asub[r_idx][c_idx] = 0.0f;
            
            // Load B
            // B is [N, K]. We load B[N_row][K_col]
            int g_n = (get_group_id(0) * TS) + r_idx;
            int g_k = tiledK + c_idx;
            if((g_n + weight_offset) < (N + weight_offset) && g_k < K) 
                Bsub[r_idx][c_idx] = B[(g_n + weight_offset) * K + g_k];
            else 
                Bsub[r_idx][c_idx] = 0.0f;
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);

        // ---------------------------------------------------------
        // 2. Compute (Outer Product in Registers)
        // ---------------------------------------------------------
        #pragma unroll
        for (int k = 0; k < TS; k++) {
            // Cache values from shared memory to registers
            float regA[WPT];
            float regB[WPT];
            
            for (int w = 0; w < WPT; w++) {
                regA[w] = Asub[col + w * RTS][k]; // Transposed access for A? No, Row-Major
                // My loading logic was linear.
                // Let's correct compute mapping:
                // Thread (row, col) computes C[globalRow + row][globalCol + col] ? No.
                // Thread computes C[... + row*RTS][... + col*RTS] ??
                
                // Correct Mapping:
                // Thread(row, col) corresponds to:
                // C rows: (get_group_id(1)*TS) + row + w*RTS
                // C cols: (get_group_id(0)*TS) + col + w*RTS
                 regA[w] = Asub[row + w * RTS][k];
                 regB[w] = Bsub[col + w * RTS][k];
            }
            
            // Compute
            for (int wa = 0; wa < WPT; wa++) {
                for (int wb = 0; wb < WPT; wb++) {
                    acc[wa][wb] += regA[wa] * regB[wb];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // ---------------------------------------------------------
    // 3. Store Results
    // ---------------------------------------------------------
    for (int w = 0; w < WPT; w++) {
        for (int r = 0; r < WPT; r++) {
            int g_r = (get_group_id(1) * TS) + row + w * RTS;
            int g_c = (get_group_id(0) * TS) + col + r * RTS;
            
            if (g_r < M && g_c < N) {
                C[g_r * N + g_c] = acc[w][r] + bias[g_c + weight_offset];
            }
        }
    }
}

// 9. GELU
__kernel void gelu_kernel(__global float* data, int total) {
    int idx = get_global_id(0);
    if (idx < total) {
        float x = data[idx];
        float val = 0.5f * x * (1.0f + erf(x * 0.70710678118f));
        
        data[idx] = val;
    }
}

// 10. Prepare Input
__kernel void prepare_class_pos_kernel(
    __global float* flat_in, __global float* enc_in,
    __global float* cls_tok, __global float* pos_emb,
    int batch_size, int embed_dim, int num_patches)
{
    int b = get_global_id(0);
    int t = get_global_id(1);
    int e = get_global_id(2);
    if (b >= batch_size || t > num_patches || e >= embed_dim) return;

    float val = (t == 0) ? cls_tok[e] : flat_in[b * num_patches * embed_dim + (t - 1) * embed_dim + e];
    enc_in[b * (num_patches + 1) * embed_dim + t * embed_dim + e] = val + pos_emb[t * embed_dim + e];
}


__kernel void extract_cls_softmax_kernel(
    __global float* src_logits, __global float* output_probs,
    int num_classes, int seq_len) 
{
    int b = get_global_id(0);
    int offset = b * seq_len * num_classes; 
    float max_val = -1e30f;
    for (int i = 0; i < num_classes; ++i) max_val = fmax(max_val, src_logits[offset + i]);
    float sum = 0.0f;
    for (int i = 0; i < num_classes; ++i) {
        float val = exp(src_logits[offset + i] - max_val);
        output_probs[b * num_classes + i] = val; 
        sum += val;
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < num_classes; ++i) output_probs[b * num_classes + i] *= inv_sum;
}

#define TILE 16

// ======================================================
// ================== FC1 (Tiled, Safe) ==================
// ======================================================
__kernel void fc1_kernel(
    __global const float* input,     // [tokens, in_features]
    __global const float* weight,    // [out_features, in_features]
    __global const float* bias,      // [out_features]
    __global float* output,          // [tokens, out_features]
    int tokens,
    int in_features,
    int out_features)
{
    int t = get_global_id(0);   // token index
    int o = get_global_id(1);   // output feature index

    int lx = get_local_id(0);   // local row
    int ly = get_local_id(1);   // local col

    // 이 워크아이템이 실제 유효한 출력 위치인지
    int valid = (t < tokens && o < out_features);

    __local float tileA[TILE][TILE];  // local tile for input
    __local float tileB[TILE][TILE];  // local tile for weight

    float sum = 0.0f;

    // iterate along in_features (K dimension)
    for (int k = 0; k < in_features; k += TILE)
    {
        // Load tile of input: A[t][k + ly]
        float a = 0.0f;
        int colA = k + ly;
        if (valid && colA < in_features)
            a = input[t * in_features + colA];
        tileA[lx][ly] = a;

        // Load tile of weight: weight[o][k + lx]
        float b = 0.0f;
        int rowB = k + lx;
        if (o < out_features && rowB < in_features)
            b = weight[o * in_features + rowB];
        tileB[lx][ly] = b;

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute partial dot-product for this tile
        for (int i = 0; i < TILE; i++)
            sum += tileA[lx][i] * tileB[i][ly];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (valid) {
        sum += bias[o];
        output[t * out_features + o] = sum;
    }
}




// ======================================================
// ================== FC2 (Tiled, Safe) ==================
// ======================================================
__kernel void fc2_kernel(
    __global const float* input,     // [tokens, hidden]
    __global const float* weight,    // [out_features, hidden]
    __global const float* bias,      // [out_features]
    __global float* output,          // [tokens, out_features]
    int tokens,
    int hidden,
    int out_features)
{
    int t = get_global_id(0);
    int o = get_global_id(1);

    int lx = get_local_id(0);
    int ly = get_local_id(1);

    int valid = (t < tokens && o < out_features);

    __local float tileA[TILE][TILE];
    __local float tileB[TILE][TILE];

    float sum = 0.0f;

    for (int k = 0; k < hidden; k += TILE)
    {
        // Load tile of input: A[t][k + ly]
        float a = 0.0f;
        int colA = k + ly;
        if (valid && colA < hidden)
            a = input[t * hidden + colA];
        tileA[lx][ly] = a;

        // Load tile of weight: weight[o][k + lx]
        float b = 0.0f;
        int rowB = k + lx;
        if (o < out_features && rowB < hidden)
            b = weight[o * hidden + rowB];
        tileB[lx][ly] = b;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int i = 0; i < TILE; i++)
            sum += tileA[lx][i] * tileB[i][ly];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (valid) {
        sum += bias[o];
        output[t * out_features + o] = sum;
    }
}
