#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>

#include "cnn_layers.cuh"
#include "kernel_functions.cuh"
#include "common.h"


Tensor *initialize_tensor() {
    Tensor *tensor = (Tensor *)malloc(sizeof(Tensor));
    tensor->num_dim = 0;
    tensor->dim = NULL;
    tensor->values_d = NULL;
    return tensor;
}


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
void run_conv2d_forward(
    Tensor *output,
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

    output->num_dim = 4;

    uint32_t *dim = (uint32_t *)malloc(output->num_dim * sizeof(uint32_t));
    dim[0] = num_samples;
    dim[1] = out_channels;
    dim[2] = out_height;
    dim[3] = out_width;

    output->dim = dim;
    output->values_d = Y_d;
}


void run_sigmoid_forward(Tensor *tensor) {
    uint32_t num_samples    = tensor->dim[0];
    uint32_t num_channels   = tensor->dim[1];
    uint32_t feature_height = tensor->dim[2];
    uint32_t feature_width  = tensor->dim[3];
    uint32_t out_size       = num_samples * num_channels * feature_height * feature_width;

    float *Y_d;
    cudaMalloc((void**)&Y_d, out_size * sizeof(float));

    uint32_t grid_height = ceil(feature_height * 1.0 / TILE_WIDTH);
    uint32_t grid_width = ceil(feature_width * 1.0 / TILE_WIDTH);
    uint32_t out_tiles_num = grid_width * grid_height;

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(num_channels, out_tiles_num, num_samples);

    SigmoidForwardKernel<<<dimGrid, dimBlock>>>(
        tensor->values_d, Y_d,
        grid_height, grid_width,
        feature_height, feature_width
    );

    // Update tensor.
    cudaFree(tensor->values_d);
    tensor->values_d = Y_d;
}


// Assume stride is always the kernel size.
void run_pooling_forward(Tensor *tensor, uint32_t kernel_length, pooling_type pool_type) {
    uint32_t num_samples    = tensor->dim[0];
    uint32_t num_channels   = tensor->dim[1];
    uint32_t feature_height = tensor->dim[2];
    uint32_t feature_width  = tensor->dim[3];
    uint32_t out_height     = feature_height / kernel_length;
    uint32_t out_width      = feature_width / kernel_length;
    uint32_t out_size       = num_samples * num_channels * out_height * out_width;

    float *Y_d;
    cudaMalloc((void**)&Y_d, out_size * sizeof(float));

    uint32_t grid_height = ceil(out_height * 1.0 / TILE_WIDTH);
    uint32_t grid_width = ceil(out_width * 1.0 / TILE_WIDTH);
    uint32_t out_tiles_num = grid_width * grid_height;

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(num_channels, out_tiles_num, num_samples);

    if (pool_type != MEAN && pool_type != MAX) {
        printf("The inputted pooling type is currently not implemented.");
        free_tensor(tensor);
        cudaFree(Y_d);
        return;
    }

    PoolForwardKernel<<<dimGrid, dimBlock>>>(
        tensor->values_d, Y_d,
        pool_type,
        kernel_length,
        grid_height, grid_width,
        feature_height, feature_width,
        out_height, out_width
    );

    // Update tensor.
    tensor->dim[2] = out_height;
    tensor->dim[3] = out_width;
    cudaFree(tensor->values_d);
    tensor->values_d = Y_d;
}
