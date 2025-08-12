#ifndef CNN_LAYERS
#define CNN_LAYERS


#include <stdint.h>


typedef struct {
    bool is_device; // True if values is from the device memory.
    uint8_t num_dim;
    uint32_t *dim;
    float *values; // Stored in row-major format.
} Tensor;


// Assume only one 1 conv and 1 linear layers.
typedef struct {
    Tensor *conv2d_weight;
    Tensor *linear_weight;
} NetworkWeights;


Tensor *initialize_conv_layer_weights(uint32_t in_channels, uint32_t out_channels, uint8_t filter_size);
Tensor *initialize_linear_layer_weights(uint32_t in_channels, uint32_t out_channels);

Tensor *run_conv2_forward(
    float *X_d,
    Tensor *filters,
    uint32_t num_samples,
    uint32_t in_height,
    uint32_t in_width
);


// CUDA Kernels.
__global__ void Conv2ForwardKernel(
    float *X, float *Y,
    Tensor *filters,
    uint32_t num_samples,
    uint32_t grid_height, uint32_t grid_width,
    uint32_t in_height, uint32_t in_width
);


#endif