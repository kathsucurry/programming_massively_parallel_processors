#ifndef CNN_LAYERS
#define CNN_LAYERS


#include <stdint.h>


typedef struct {
    uint8_t num_dim;
    uint32_t *dim;
    float *values_d; // Stored in device + row-major format.
} Tensor;


// Assume only one 1 conv and 1 linear layers.
typedef struct {
    Tensor *conv2d_weight;
    Tensor *linear_weight;
} NetworkWeights;


Tensor *initialize_conv_layer_weights(uint32_t in_channels, uint32_t out_channels, uint8_t filter_size, uint32_t seed);
Tensor *initialize_linear_layer_weights(uint32_t in_channels, uint32_t out_channels, uint32_t seed);
uint32_t get_tensor_values_size(const uint8_t num_dim, const uint32_t *dim);
void free_tensor(Tensor *tensor);

Tensor *run_conv2d_forward(
    float *X_d,
    Tensor *filters,
    uint32_t num_samples,
    uint32_t in_height,
    uint32_t in_width
);


// CUDA Kernels.
__global__ void Conv2ForwardKernel(
    float *X, float *Y,
    float *filters,
    uint32_t kernel_length,
    uint32_t in_channels,
    uint32_t grid_height, uint32_t grid_width,
    uint32_t in_height, uint32_t in_width,
    uint32_t out_height, uint32_t out_width
);



#endif