#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>

#include "cnn_layers.cuh"
#include "common.h"


#define TILE_WIDTH 32


float *_uniform_xavier_initialization(uint32_t fan_in, uint32_t fan_out, uint32_t size, uint32_t seed) {
    // Assume gain = 1.
    srand(seed);
    float x = sqrtf(6.0 / (fan_in + fan_out));
    float *array = (float *)malloc(size * sizeof(float));
    for (uint32_t i = 0; i < size; ++i)
        array[i] = x * 2 * (rand() * 1.0 / RAND_MAX) - x; 
    return array;
}


// For simplicity, assume stride is always 1.
Tensor *initialize_conv_layer_weights(
    uint32_t in_channels,
    uint32_t out_channels,
    uint8_t filter_length,
    uint32_t seed
) {
    Tensor *conv_weight = (Tensor *)malloc(sizeof(Tensor));
    // Dimensions = out_channels * in_channels * filter_length * filter_length.
    conv_weight->num_dim = 4;
    uint32_t *dim = (uint32_t *)malloc(conv_weight->num_dim * sizeof(uint32_t));
    dim[0] = out_channels;
    dim[1] = in_channels;
    dim[2] = filter_length;
    dim[3] = filter_length;
    conv_weight->dim = dim;

    uint32_t weight_size = out_channels * in_channels * filter_length * filter_length;
    uint32_t fan_in = in_channels * filter_length * filter_length;
    uint32_t fan_out = out_channels * filter_length * filter_length;

    float *filters = _uniform_xavier_initialization(fan_in, fan_out, weight_size, seed);
    float *filters_d;
    cudaMalloc((void**)&filters_d, weight_size * sizeof(float));
    cudaMemcpy(filters_d, filters, weight_size * sizeof(float), cudaMemcpyHostToDevice);
    conv_weight->values_d = filters_d;
    free(filters);
    
    return conv_weight;
}


Tensor *initialize_linear_layer_weights(uint32_t in_channels, uint32_t out_channels, uint32_t seed) {
    Tensor *linear_weight = (Tensor *)malloc(sizeof(Tensor));
    linear_weight->num_dim = 2;

    uint32_t *dim = (uint32_t *)malloc(linear_weight->num_dim * sizeof(uint32_t));
    dim[0] = out_channels;
    dim[1] = in_channels;
    linear_weight->dim = dim;
    uint32_t weight_size = out_channels * in_channels;

    float *weights = _uniform_xavier_initialization(in_channels, out_channels, weight_size, seed);
    float *weights_d;
    cudaMalloc((void**)&weights_d, weight_size * sizeof(float));
    cudaMemcpy(weights_d, weights, weight_size * sizeof(float), cudaMemcpyHostToDevice);
    linear_weight->values_d = weights_d;
    free(weights);

    return linear_weight;
}


uint32_t get_tensor_values_size(const uint8_t num_dim, const uint32_t *dim) {
    uint32_t size = 1;
    for (uint8_t i = 0; i < num_dim; ++i)
        size *= dim[i];
    return size;
}


void free_tensor(Tensor *tensor) {
    cudaFree(tensor->values_d);
    free(tensor->dim);
    free(tensor);
}


/**
 * Conv2 kernel implementation, following the tiled method in chapter 16.3
 */
Tensor *run_conv2d_forward(
    float *X_d,
    Tensor *filters,
    uint32_t num_samples,
    uint32_t in_height,
    uint32_t in_width
) {
    float *Y_d;

    uint32_t filter_length = filters->dim[filters->num_dim - 2];
    uint32_t out_height    = in_height - filter_length + 1;
    uint32_t out_width     = in_width - filter_length + 1;
    uint32_t out_channels  = filters->dim[0];
    uint32_t in_channels   = filters->dim[1];
    uint32_t out_size      = num_samples * out_channels * out_height * out_width;

    cudaMalloc((void**)&Y_d, out_size * sizeof(float));

    uint32_t grid_width = ceil(out_width * 1.0 / TILE_WIDTH);
    uint32_t grid_height = ceil(out_height * 1.0 / TILE_WIDTH);
    uint32_t out_tiles_num = grid_width * grid_height;

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid(out_channels, out_tiles_num, num_samples);
    Conv2ForwardKernel<<<dimGrid, dimBlock>>>(
        X_d, Y_d,
        filters->values_d,
        filter_length,
        in_channels,
        grid_height, grid_width,
        in_height, in_width,
        out_height, out_width
    );

    Tensor *output = (Tensor *)malloc(sizeof(Tensor));
    output->num_dim = 4;

    uint32_t *dim = (uint32_t *)malloc(output->num_dim * sizeof(uint32_t));
    dim[0] = num_samples;
    dim[1] = out_channels;
    dim[2] = out_height;
    dim[3] = out_width;

    output->dim = dim;
    output->values_d = Y_d;

    return output;
}


/**
 * Conv2 kernel implementation, following the method in chapter 16.3 (Fig. 16.13,14).
 */
__global__ void Conv2ForwardKernel(
    float *X, float *Y,
    float *filters,
    uint32_t kernel_length,
    uint32_t in_channels,
    uint32_t grid_height, uint32_t grid_width,
    uint32_t in_height, uint32_t in_width,
    uint32_t out_height, uint32_t out_width
) {
    uint32_t out_channel_idx = blockIdx.x;
    uint32_t out_height_idx  = (blockIdx.y / grid_width)*TILE_WIDTH + threadIdx.y;
    uint32_t out_width_idx   = (blockIdx.y % grid_width)*TILE_WIDTH + threadIdx.x;
    uint32_t sample_idx      = blockIdx.z;
    uint32_t out_channels = gridDim.x;

    if (out_height_idx >= out_height || out_width_idx >= out_width)
        return;
    
    float value = 0.0f;
    for (uint32_t in_channel_idx = 0; in_channel_idx < in_channels; ++in_channel_idx)
        for (uint32_t k_row = 0; k_row < kernel_length; ++k_row)
            for (uint32_t k_col = 0; k_col < kernel_length; ++k_col) {
                uint32_t in_row = out_height_idx + k_row;
                uint32_t in_col = out_width_idx + k_col;
                
                uint32_t X_idx = (sample_idx * in_channels * in_height * in_width) + 
                    (in_channel_idx * in_height * in_width) + 
                    (in_row * in_width) + 
                    in_col;
                uint32_t weight_idx = (out_channel_idx * in_channels * kernel_length * kernel_length) +
                    (in_channel_idx * kernel_length * kernel_length) +
                    (k_row * kernel_length) +
                    k_col;
                value += X[X_idx] * filters[weight_idx];
            }
    
    uint32_t Y_idx = (sample_idx * out_channels * out_height * out_width) + 
            (out_channel_idx * out_height * out_width) +
            (out_height_idx * out_width) +
            out_width_idx;
    Y[Y_idx] = value;
}