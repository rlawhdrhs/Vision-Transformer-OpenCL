// fc1_kernel.cl
__kernel void fc1_kernel(
    __global const float* input,   // [tokens, in_features]
    __global const float* weight,  // [out_features, in_features] (row-major)
    __global const float* bias,    // [out_features]
    __global float* output,        // [tokens, out_features]
    int tokens,
    int in_features,
    int out_features)
{
    int t = get_global_id(0); // token index
    int o = get_global_id(1); // output feature index

    if (t >= tokens || o >= out_features) return;

    int in_offset = t * in_features;
    int w_offset  = o * in_features;

    float sum = bias[o];
    for (int i = 0; i < in_features; i++) {
        sum += input[in_offset + i] * weight[w_offset + i];
    }

    output[t * out_features + o] = sum;
}
