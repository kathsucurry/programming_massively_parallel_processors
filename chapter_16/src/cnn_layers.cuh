#ifndef CNN_LAYERS
#define CNN_LAYERS


#include <stdint.h>


// Exclude bias for simplicity.
typedef struct {
    uint32_t in_channels;
    uint32_t out_channels;
    uint8_t filter_size;
    float *filters_d; // Learnable weights.
} Conv2DLayer;

typedef struct {
    uint32_t in_channels;
    uint32_t out_channels;
    float *weights_d;
} LinearLayer;

typedef struct {
    Conv2DLayer* conv2d;
    LinearLayer* linear;
} NetworkWeights;

Conv2DLayer *initialize_conv_layer_weights(uint32_t in_channels, uint32_t out_channels, uint8_t filter_size);
LinearLayer *initialize_linear_layer_weights(uint32_t in_channels, uint32_t out_channels);



// CUDA Kernels.
__global__ void run_conv2_forward();

#endif