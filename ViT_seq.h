#include "Network.h"


#ifndef _ViT_seq_H
#define _ViT_seq_H
void Conv2d(float *input, float *output, Network weight, Network bias);
void flatten_transpose(float *input, float *output);
void class_token(float *patch_tokens, float *final_tokens, Network cls_tk);
void pos_emb(float *input, float *output, Network pos_emb);
void layer_norm(float *input, float *output, Network weight, Network bias);
void multihead_attn(float *input, float *output, Network in_weight, Network in_bias, Network out_weight, Network out_bias);
float gelu(float x);
void gelu_activation(float *input, float *output, int size);
void linear_layer(float *input, float *output, int tokens, int in_features, int out_features, Network weight, Network bias);
void mlp_block(float *input, float *output, Network fc1_weight, Network fc1_bias, Network fc2_weight, Network fc2_bias);
void Encoder(float *input, float *output,
    Network ln1_w, Network ln1_b, Network attn_w, Network attn_b, Network attn_out_w, Network attn_out_b,
    Network ln2_w, Network ln2_b, Network mlp1_w, Network mlp1_b, Network mlp2_w, Network mlp2_b);
void Softmax(float *logits, float *probabilities, int length);
void ViT_seq(ImageData *image, Network *networks, float **prb);
const int size[];
const int enc_size;

#endif#pragma once