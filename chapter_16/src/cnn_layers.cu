#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>

#include "cnn_layers.cuh"


float *_uniform_xavier_initialization(uint32_t fan_in, uint32_t fan_out, uint32_t size) {
    // Assume gain = 1.
    float x = sqrtf(6.0 / (fan_in + fan_out));
    float *array = (float *)malloc(size * sizeof(float));
    for (uint32_t i = 0; i < size; ++i)
        array[i] = x * 2 * (rand() * 1.0 / RAND_MAX) - x; 
    return array;
}

// For simplicity, assume stride is always 1.
Conv2DLayer *initialize_conv_layer_weights(
    uint32_t in_channels,
    uint32_t out_channels,
    uint8_t filter_size
) {
    Conv2DLayer *conv = (Conv2DLayer *)malloc(sizeof(Conv2DLayer));
    conv->in_channels = in_channels;
    conv->out_channels = out_channels;
    conv->filter_size = filter_size;

    uint32_t fan_in = in_channels * filter_size * filter_size;
    uint32_t fan_out = out_channels * filter_size * filter_size;

    float *filters = _uniform_xavier_initialization(fan_in, fan_out, fan_out);
    float *filters_d;
    cudaMalloc((void**)&filters_d, fan_out * sizeof(float));
    cudaMemcpy(filters_d, filters, fan_out * sizeof(float), cudaMemcpyHostToDevice);
    conv->filters_d = filters_d;

    free(filters);
    return conv;
}


LinearLayer *initialize_linear_layer_weights(uint32_t in_channels, uint32_t out_channels) {
    LinearLayer *linear = (LinearLayer *)(sizeof(LinearLayer));
    linear->in_channels = in_channels;
    linear->out_channels = out_channels;

    float *weights = _uniform_xavier_initialization(in_channels, out_channels, out_channels * in_channels);
    float *weights_d;

    cudaMalloc((void**)&weights_d, out_channels * in_channels * sizeof(float));
    cudaMemcpy(weights_d, weights, out_channels * in_channels * sizeof(float), cudaMemcpyHostToDevice);
    linear->weights_d = weights_d;
    return linear;
}


__global__ void run_conv2_forward() {

}