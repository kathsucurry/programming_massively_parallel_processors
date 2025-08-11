#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#include "cnn_layers.cuh"


float *_uniform_xavier_initialization_1d(uint32_t fan_in, uint32_t fan_out, uint8_t size) {
    // Assume gain = 1.
    float x = sqrtf(6.0 / (fan_in + fan_out));
    float *array = (float *)malloc(size * sizeof(float));
    for (uint8_t i = 0; i < size; ++i)
        array[i] = x * 2 * (rand() * 1.0 / RAND_MAX) - x; 
    return array;
}


float **_uniform_xavier_initialization_2d(uint32_t fan_in, uint32_t fan_out, uint8_t height, uint8_t width) {
    // Assume gain = 1.
    float x = sqrtf(6.0 / (fan_in + fan_out));

    float **array = (float **)malloc(height * sizeof(float));
    for (uint8_t row = 0; row < height; ++row) {
        array[row] = (float *)malloc(width * sizeof(float));
        for (uint8_t col = 0; col < width; ++col) {
            array[row][col] = x * 2 * (rand() * 1.0 / RAND_MAX) - x; 
        }
    }
    return array;
}


// For simplicity, assume stride is always 1.
Conv2DLayerWeights *initialize_conv_layer_weights(
    uint8_t in_channels,
    uint8_t out_channels,
    uint8_t filter_size
) {
    Conv2DLayerWeights *conv = (Conv2DLayerWeights *)(sizeof(Conv2DLayerWeights));

    uint32_t fan_in = in_channels * filter_size * filter_size;
    uint32_t fan_out = out_channels * filter_size * filter_size;
    conv->filters = _uniform_xavier_initialization_2d(fan_in, fan_out, filter_size, filter_size);
    conv->filter_size = filter_size;
    conv->num_filters = out_channels;
    return conv;
}


LinearLayerWeights *initialize_linear_layer_weights(uint8_t in_channels, uint8_t out_channels) {
    LinearLayerWeights *linear = (LinearLayerWeights *)(sizeof(LinearLayerWeights));
    linear->in_channels = in_channels;
    linear->out_channels = out_channels;
    linear->weights = _uniform_xavier_initialization_2d(in_channels, out_channels, out_channels, in_channels);
    return linear;
}


