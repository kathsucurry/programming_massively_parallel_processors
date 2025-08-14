#ifndef CNN_LAYERS
#define CNN_LAYERS


#include <stdint.h>

#include "kernel_functions.cuh"

typedef struct {
    uint8_t num_dim;
    uint32_t *dim;
    float *values_d; // Stored in device + row-major format.
} Tensor;


// Assume only one 1 conv and 1 linear layers for now.
typedef struct {
    Tensor *conv2d_weight;
    Tensor *linear_weight;
} NetworkWeights;


Tensor *initialize_tensor();
Tensor *initialize_conv_layer_weights(uint32_t in_channels, uint32_t out_channels, uint8_t filter_size, uint32_t seed);
Tensor *initialize_linear_layer_weights(uint32_t in_channels, uint32_t out_channels, uint32_t seed);
uint32_t get_tensor_values_size(const uint8_t num_dim, const uint32_t *dim);
void free_tensor(Tensor *tensor);

/* Forward layer functions */
void run_conv2d_forward(
    Tensor *output,
    float *X_d,
    Tensor *filters,
    uint32_t num_samples,
    uint32_t in_height,
    uint32_t in_width
);


void run_sigmoid_forward(Tensor *tensor);

void run_pooling_forward(Tensor *tensor, uint32_t kernel_length, pooling_type pool_type);

void run_flatten_layer(Tensor *tensor);


#endif