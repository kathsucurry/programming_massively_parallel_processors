#ifndef CNN_LAYERS
#define CNN_LAYERS


#include <stdint.h>


// Exclude bias for simplicity.
typedef struct {
    uint8_t num_filters;
    float **filters;
    uint8_t filter_size;
} Conv2DLayerWeights;

typedef struct {
    uint8_t in_channels;
    uint8_t out_channels;
    float **weights;
} LinearLayerWeights;

typedef struct {
    Conv2DLayerWeights* conv2_weights;
    LinearLayerWeights* linear_weights;
} CNNNetworkWeights;

float *_uniform_xavier_initialization_1d(uint32_t fan_in, uint32_t fan_out, uint8_t size);
float **_uniform_xavier_initialization_2d(uint32_t fan_in, uint32_t fan_out, uint8_t height, uint8_t width);
Conv2DLayerWeights *initialize_conv_layer(uint8_t in_channels, uint8_t out_channels, uint8_t num_filters, uint8_t filter_size);
LinearLayerWeights *initialize_linear_layer(uint8_t in_channels, uint8_t out_channels);


#endif