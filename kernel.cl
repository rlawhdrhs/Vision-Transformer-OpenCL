#define PATCH_SIZE 16
#define IN_CHANS 3
#define EMBED_DIM 768
#define IMG_SIZE 224
#define OUT_SIZE (IMG_SIZE / PATCH_SIZE) 
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
/*
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
*/

__kernel void Conv2d_Batched_Kernel(
    __global float* input, __global float* output,
    __global float* weight, __global float* bias
) 
{
    int oc = get_global_id(0);
    int oh = get_global_id(1);
    int global_z = get_global_id(2);

    int batch_idx = global_z / OUT_SIZE;
    int ow = global_z % OUT_SIZE;

    if (oc >= EMBED_DIM || oh >= OUT_SIZE) return;

    int in_batch_off = batch_idx * IN_CHANS * IMG_SIZE * IMG_SIZE;
    int out_batch_off = batch_idx * EMBED_DIM * OUT_SIZE * OUT_SIZE;
    
    float sum = bias[oc];
    int w_oc_off = oc * IN_CHANS * PATCH_SIZE * PATCH_SIZE;

    for (int ic = 0; ic < IN_CHANS; ++ic) { 
        
        int in_ic_off = in_batch_off + ic * IMG_SIZE * IMG_SIZE;
        int w_ic_off = w_oc_off + ic * PATCH_SIZE * PATCH_SIZE;
        
        for (int kh = 0; kh < PATCH_SIZE; ++kh) {
            int in_ih_off = in_ic_off + (oh * PATCH_SIZE + kh) * IMG_SIZE;
            int w_kh_off = w_ic_off + kh * PATCH_SIZE;
            
            for (int kw = 0; kw < PATCH_SIZE; ++kw) {
                sum += input[in_ih_off + ow * PATCH_SIZE + kw] * weight[w_kh_off + kw];
            }
        }
    }
    output[out_batch_off + (oc * OUT_SIZE + oh) * OUT_SIZE + ow] = sum;
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
            if (gi < K && tj < N) Bsub[j][i] = B[B_off + gi * B_stride + tj];
            else Bsub[j][i] = 0.0f;
        } else {
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
    // Global ID
    int batch_head = get_group_id(2); // Batch * NumHeads
    int row = get_global_id(1);       // Token Q (M)
    int col = get_global_id(0);       // Token K (N)

    // Local ID
    int local_row = get_local_id(1);
    int local_col = get_local_id(0);

    // Local Memory (Shared)
    __local float Q_tile[TS][TS + 1]; // Bank Conflict 방지 패딩
    __local float K_tile[TS][TS + 1];

    int batch_idx = batch_head / num_heads;
    int head_idx  = batch_head % num_heads;
    
    // Offset 계산
    // Q, K 구조: [Batch, Tokens, EmbedDim] (EmbedDim = NumHeads * HeadDim)
    int embed_dim = num_heads * head_dim;
    int base_offset = batch_idx * tokens * embed_dim + head_idx * head_dim;

    float sum = 0.0f;
    int num_tiles = (head_dim + TS - 1) / TS;

    for (int t = 0; t < num_tiles; t++) {
        int tiled_d = t * TS;

        int q_d = tiled_d + local_col;
        if (row < tokens && q_d < head_dim)
            Q_tile[local_row][local_col] = Q[base_offset + row * embed_dim + q_d];
        else
            Q_tile[local_row][local_col] = 0.0f;

        int k_d = tiled_d + local_row; // Transposed loading trick
        
        int k_token = col; 
        int k_dim   = tiled_d + local_row; // Loop variable match
        

        if (col < tokens && (tiled_d + local_row) < head_dim)
            K_tile[local_col][local_row] = K[base_offset + col * embed_dim + (tiled_d + local_row)];
        else
            K_tile[local_col][local_row] = 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TS; k++) {
            sum += Q_tile[local_row][k] * K_tile[local_col][k]; // K는 위에서 Transpose해서 로드함
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < tokens && col < tokens) {
        // Scores: [Batch, NumHeads, Tokens, Tokens]
        int score_idx = (batch_head * tokens * tokens) + (row * tokens) + col;
        scores[score_idx] = sum * scale;
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
    // Global ID
    int batch_head = get_group_id(2);
    int row = get_global_id(1); // Output Token Index (Row of Score)
    int col = get_global_id(0); // Head Dim Index (Col of V)

    // Local ID
    int local_row = get_local_id(1);
    int local_col = get_local_id(0);

    __local float S_tile[TS][TS + 1];
    __local float V_tile[TS][TS + 1];

    int batch_idx = batch_head / num_heads;
    int head_idx  = batch_head % num_heads;

    int embed_dim = num_heads * head_dim;
    
    // Offsets
    int score_base = batch_head * tokens * tokens;
    int v_base = batch_idx * tokens * embed_dim + head_idx * head_dim;
    int out_base = batch_idx * tokens * embed_dim + head_idx * head_dim; // V와 동일 구조

    float sum = 0.0f;
    int num_tiles = (tokens + TS - 1) / TS;

    for (int t = 0; t < num_tiles; t++) {
        int tiled_k = t * TS; // Common dimension (Tokens)

        // 1. Load Score Tile [Row, K]
        int s_k = tiled_k + local_col;
        if (row < tokens && s_k < tokens)
            S_tile[local_row][local_col] = scores[score_base + row * tokens + s_k];
        else
            S_tile[local_row][local_col] = 0.0f;

        // 2. Load V Tile [K, Col]
        int v_k = tiled_k + local_row;
        if (v_k < tokens && col < head_dim)
            V_tile[local_row][local_col] = V[v_base + v_k * embed_dim + col];
        else
            V_tile[local_row][local_col] = 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        // 3. Compute
        for (int k = 0; k < TS; k++) {
            sum += S_tile[local_row][k] * V_tile[k][local_col];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store
    if (row < tokens && col < head_dim) {
        // Output: [Batch, Tokens, EmbedDim] (Interleaved heads)
        output[out_base + row * embed_dim + col] = sum;
    }
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

// 8. Linear
__kernel void linear_kernel(
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
        
        for (int i = 0; i < 4; i++) {
             int c = tiledCol + i;
             if (globalRow < M && c < K)
                 Asub[row][col * 4 + i] = A[globalRow * K + c];
             else
                 Asub[row][col * 4 + i] = 0.0f;
        }
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

__kernel __attribute__((reqd_work_group_size(TS / 4, TS, 1)))
__attribute__((vec_type_hint(float4)))
void linear_kernel_opt(
    int M, int N, int K,
    __global const float* A,
    __global const float* B,
    __global const float* bias,
    __global float* C,
    int weight_offset) 
{
    // Thread ID
    const int local_col = get_local_id(0);
    const int local_row = get_local_id(1);
    const int group_col = get_group_id(0);
    const int group_row = get_group_id(1);

    // Global Output Coordinate
    const int col = (group_col * TS) + (local_col * 4);
    const int row = (group_row * TS) + local_row;

    // Local Memory
    __local float Asub[TS][TS + 1];
    __local float Bsub[TS][TS + 1];

    float4 acc = (float4)(0.0f);

    // Loop over tiles
    const int numTiles = (K + TS - 1) / TS;
    
    const int local_idx = local_row * (TS/4) + local_col; // 0 .. 255
    
    for (int t = 0; t < numTiles; t++) {
        const int tiledK = t * TS;

        // 1. Load A [Row, K]
        for(int i=0; i<4; i++) {
             int tiled_k_idx = t*TS + local_col*4 + i;
        }

        // --- B 행렬 로딩 ---
        
        for (int i = 0; i < 4; i++) {
            // 로드할 B 내부 좌표 (tr, tc)
            int flat_idx = local_idx * 4 + i;
            int tr = flat_idx / TS;
            int tc = flat_idx % TS;
            
            int global_b_row = (group_col * TS) + tr + weight_offset;
            int global_b_col = tiledK + tc;

            float val = 0.0f;
            if (global_b_row < (N + weight_offset) && global_b_col < K) {
                val = B[global_b_row * K + global_b_col];
            }
            
            Bsub[tc][tr] = val; 
        }

        // A 로딩
        for (int i = 0; i < 4; i++) {
            int flat_idx = local_idx * 4 + i;
            int tr = flat_idx / TS;
            int tc = flat_idx % TS;
            
            int global_a_row = (group_row * TS) + tr;
            int global_a_col = tiledK + tc;
            
            float val = 0.0f;
            if (global_a_row < M && global_a_col < K) {
                val = A[global_a_row * K + global_a_col];
            }
            Asub[tr][tc] = val;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TS; k++) {
            float valA = Asub[local_row][k];
            float4 valB;
            valB.x = Bsub[k][local_col * 4 + 0];
            valB.y = Bsub[k][local_col * 4 + 1];
            valB.z = Bsub[k][local_col * 4 + 2];
            valB.w = Bsub[k][local_col * 4 + 3];

            acc += valA * valB;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store Result
    if (row < M && col < N) {
        // Bias Loading
        float4 b_val = (float4)(0.0f);
        if (col + 3 < N) { // Boundary check safe logic required ideally
             b_val.x = bias[col + 0 + weight_offset];
             b_val.y = bias[col + 1 + weight_offset];
             b_val.z = bias[col + 2 + weight_offset];
             b_val.w = bias[col + 3 + weight_offset];
        }
        
        // Vector store
        __global float4* C_ptr = (__global float4*)&C[row * N + col];
        *C_ptr = acc + b_val;
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
