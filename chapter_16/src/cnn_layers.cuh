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

typedef struct {
    Tensor *dW_or_W;
    Tensor *dX_or_X;
    bool is_grad; // True if dW/dX is stored, else it saves W/X for later computation in the chain rule.
} LayerGradients;

typedef struct {
    LayerGradients *gradients;
    uint32_t num_layers;
    Tensor *output;
} NetworkOutputs;


typedef struct {
    float loss;
    float accuracy_percent;
} EpochOutput;


uint32_t get_tensor_values_size(const uint8_t num_dim, const uint32_t *dim);
Tensor *initialize_tensor(float *X, uint8_t num_dim, uint32_t *dim);
Tensor *deep_copy_tensor(Tensor *tensor);
void free_tensor(Tensor *tensor);
void free_layer_gradients(LayerGradients *gradients);
void free_network_weights(NetworkWeights *weights);
void free_network_outputs(NetworkOutputs *output, bool include_grad);

Tensor *initialize_conv_layer_weights(uint32_t in_channels, uint32_t out_channels, uint8_t filter_size, uint32_t seed);
Tensor *initialize_linear_layer_weights(uint32_t in_channels, uint32_t out_channels, uint32_t seed);

/* Forward layer functions */

void run_conv2d_forward(Tensor *output, Tensor *filters, LayerGradients *grad, bool compute_grad);
void run_conv2d_backward(Tensor *conv2d_weights, LayerGradients *grad, LayerGradients *next_layer_grad, float learning_rate);

void run_sigmoid_forward(Tensor *tensor, LayerGradients *grad, bool compute_grad);
void run_sigmoid_backward(LayerGradients *grad, LayerGradients *next_layer_grad);

void run_pooling_forward(Tensor *tensor, uint32_t kernel_length, pooling_type pool_type, LayerGradients *grad, bool compute_grad);
void run_pooling_backward(uint32_t kernel_length, LayerGradients *grad, LayerGradients *next_layer_grad);

void run_flatten_forward(Tensor *tensor);
void run_flatten_backward(uint32_t num_samples, uint8_t kernel_length, LayerGradients *grad, LayerGradients *next_layer_grad);

void run_linear_forward(Tensor *X, Tensor *linear_weights, LayerGradients *grad, bool compute_grad);
void run_linear_backward(Tensor *linear_weights, LayerGradients *grad, LayerGradients *next_layer_grad, float lr);

void run_softmax_forward(Tensor *tensor, uint8_t *y_d, LayerGradients *grad, bool compute_grad);

float *compute_negative_log_likelihood_log_lost(Tensor *tensor, uint8_t *y_d);

#endif
