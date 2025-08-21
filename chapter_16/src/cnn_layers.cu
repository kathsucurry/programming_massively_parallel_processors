#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>

#include "cnn_layers.cuh"
#include "kernel_functions.cuh"
#include "common.h"


uint32_t get_tensor_values_size(const uint8_t num_dim, const uint32_t *dim) {
    uint32_t size = 1;
    for (uint8_t i = 0; i < num_dim; ++i)
        size *= dim[i];
    return size;
}


Tensor *initialize_tensor(float *X, uint8_t num_dim, uint32_t *dim) {
    Tensor *tensor = (Tensor *)malloc(sizeof(Tensor));
    tensor->num_dim = num_dim;
    tensor->dim = dim;

    uint32_t size = get_tensor_values_size(num_dim, dim);
    float *values_d;
    cudaMalloc((void**)&values_d, size * sizeof(float));
    cudaMemcpy(values_d, X, size * sizeof(float), cudaMemcpyHostToDevice);

    tensor->values_d = values_d;
    return tensor;
}


Tensor *deep_copy_tensor(Tensor *tensor) {
    Tensor *new_tensor = (Tensor *)malloc(sizeof(Tensor));
    new_tensor->num_dim = tensor->num_dim;

    new_tensor->dim = (uint32_t *)malloc(new_tensor->num_dim * sizeof(uint32_t));
    memcpy(new_tensor->dim, tensor->dim, new_tensor->num_dim * sizeof(uint32_t));

    uint32_t out_size = get_tensor_values_size(new_tensor->num_dim, new_tensor->dim);
    float *new_tensor_values_d;
    cudaMalloc((void**)&new_tensor_values_d, out_size * sizeof(float));
    cudaMemcpy(new_tensor_values_d, tensor->values_d, out_size * sizeof(float), cudaMemcpyDeviceToDevice);

    new_tensor->values_d = new_tensor_values_d;
    return new_tensor;
}


void free_tensor(Tensor *tensor) {
    if (tensor == NULL)
        return;
    cudaFree(tensor->values_d);
    free(tensor->dim);
    free(tensor);
}


void free_layer_gradients(LayerGradients *gradients) {
    free_tensor(gradients->dW_or_W);
    free_tensor(gradients->dX_or_X);
    free(gradients);
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
    conv_weight->dim = (uint32_t *)malloc(conv_weight->num_dim * sizeof(uint32_t));
    conv_weight->dim[0] = out_channels;
    conv_weight->dim[1] = in_channels;
    conv_weight->dim[2] = filter_length;
    conv_weight->dim[3] = filter_length;

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


Tensor *initialize_linear_layer_weights(uint32_t in_features, uint32_t out_features, uint32_t seed) {
    Tensor *linear_weight = (Tensor *)malloc(sizeof(Tensor));
    linear_weight->num_dim = 2;

    linear_weight->dim = (uint32_t *)malloc(linear_weight->num_dim * sizeof(uint32_t));
    linear_weight->dim[0] = out_features;
    linear_weight->dim[1] = in_features;
    uint32_t weight_size = out_features * in_features;

    float *weights = _uniform_xavier_initialization(in_features, out_features, weight_size, seed);
    float *weights_d;
    cudaMalloc((void**)&weights_d, weight_size * sizeof(float));
    cudaMemcpy(weights_d, weights, weight_size * sizeof(float), cudaMemcpyHostToDevice);
    linear_weight->values_d = weights_d;
    free(weights);

    return linear_weight;
}


/**
 * Conv2 kernel implementation, following the tiled method in chapter 16.3
 */
void run_conv2d_forward(Tensor *output, Tensor *filters, LayerGradients *grad) {
    uint32_t num_samples = output->dim[0];
    uint32_t in_channels = output->dim[1];
    uint32_t in_height   = output->dim[2];
    uint32_t in_width    = output->dim[3];

    uint32_t filter_length = filters->dim[filters->num_dim - 1];
    uint32_t out_height    = in_height - filter_length + 1;
    uint32_t out_width     = in_width - filter_length + 1;
    uint32_t out_channels  = filters->dim[0];
    uint32_t out_size      = num_samples * out_channels * out_height * out_width;

    // Store tensors for backprop later.    
    grad->dW_or_W = NULL;
    grad->dX_or_X = deep_copy_tensor(output);
    grad->is_grad = false;

    float *Y_d;
    cudaMalloc((void**)&Y_d, out_size * sizeof(float));

    uint32_t grid_width = ceil(out_width * 1.0 / TILE_WIDTH);
    uint32_t grid_height = ceil(out_height * 1.0 / TILE_WIDTH);
    uint32_t out_tiles_num = grid_width * grid_height;

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid(out_channels, out_tiles_num, num_samples);
    Conv2DForwardKernel<<<dimGrid, dimBlock>>>(
        output->values_d, Y_d,
        filters->values_d,
        filter_length,
        in_channels,
        grid_height, grid_width,
        in_height, in_width
    );

    // Update output tensor.
    output->num_dim = 4;
    free(output->dim);
    output->dim = (uint32_t *)malloc(output->num_dim * sizeof(uint32_t));
    output->dim[0] = num_samples;
    output->dim[1] = out_channels;
    output->dim[2] = out_height;
    output->dim[3] = out_width;
    output->values_d = Y_d;
}


void run_conv2d_backward(Tensor *conv2d_weights, LayerGradients *grad, LayerGradients *next_layer_grad, float lr) {
    // Recall that in grad stores W in dW and X in dX (i.e., not gradient values).
    // num samples, out channels, filter length
    Tensor *dY = next_layer_grad->dX_or_X;
    Tensor *X  = grad->dX_or_X;
    uint32_t num_samples   = X->dim[0];
    uint32_t in_channels   = conv2d_weights->dim[1];
    uint32_t out_channels  = conv2d_weights->dim[0];
    uint32_t filter_length = conv2d_weights->dim[conv2d_weights->num_dim - 1]; 
    uint32_t in_height     = X->dim[2];
    uint32_t in_width      = X->dim[3];
    uint32_t out_height    = in_height - filter_length + 1;
    uint32_t out_width     = in_width - filter_length + 1;
    uint32_t in_size       = num_samples * in_channels * in_height * in_width;
    uint32_t weight_size   = out_channels * in_channels * filter_length * filter_length;

    // Calculate dX.
    float *dX_d;
    cudaMalloc((void**)&dX_d, in_size * sizeof(float));
    cudaMemset(dX_d, 0, in_size * sizeof(float));

    uint32_t grid_width = ceil(in_width * 1.0 / TILE_WIDTH);
    uint32_t grid_height = ceil(in_height * 1.0 / TILE_WIDTH);
    uint32_t out_tiles_num = grid_width * grid_height;

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGriddX(in_channels, out_tiles_num, num_samples);
    Conv2DBackwardXGradKernel<<<dimGriddX, dimBlock>>>(
        dX_d, dY->values_d, conv2d_weights->values_d,
        filter_length,
        out_channels,
        grid_height, grid_width,
        in_height, in_width
    );

    Tensor *dX = (Tensor *)malloc(sizeof(Tensor));
    dX->values_d = dX_d;
    dX->num_dim  = 4;
    dX->dim      = (uint32_t *)malloc(4 * sizeof(uint32_t));
    memcpy(dX->dim, X->dim, 4 * sizeof(uint32_t));

    // Calculate dW.
    float *dW_d;
    cudaMalloc((void**)&dW_d, weight_size * sizeof(float));
    cudaMemset(dW_d, 0, weight_size * sizeof(float));

    grid_width  = ceil(filter_length * 1.0 / TILE_WIDTH);
    grid_height = ceil(filter_length * 1.0 / TILE_WIDTH);
    out_tiles_num = grid_width * grid_height;

    dim3 dimGriddW(in_channels, out_tiles_num, out_channels);
    Conv2DBackwardWGradKernel<<<dimGriddW, dimBlock>>>(
        dW_d, dY->values_d, X->values_d,
        filter_length,
        num_samples,
        grid_height, grid_width,
        out_height, out_width
    );

    Tensor *dW = (Tensor *)malloc(sizeof(Tensor));
    dW->values_d = dW_d;
    dW->num_dim  = 4;
    dW->dim      = (uint32_t *)malloc(4 * sizeof(uint32_t));
    memcpy(dW->dim, conv2d_weights->dim, 4 * sizeof(uint32_t));

    // Update W.
    Update3DGridParameterKernel<<<dimGriddW, dimBlock>>>(
        conv2d_weights->values_d, dW_d, filter_length, filter_length, grid_height, grid_width, lr
    );
    
    // Update grad.
    free_tensor(X);
    grad->dX_or_X = dX;
    grad->dW_or_W = dW;
    grad->is_grad = true;
}


void run_sigmoid_forward(Tensor *tensor, LayerGradients *grad) {
    uint32_t num_samples    = tensor->dim[0];
    uint32_t num_channels   = tensor->dim[1];
    uint32_t feature_height = tensor->dim[2];
    uint32_t feature_width  = tensor->dim[3];
    uint32_t out_size       = num_samples * num_channels * feature_height * feature_width;

    float *Y_d;
    cudaMalloc((void**)&Y_d, out_size * sizeof(float));

    // Prepare gradients.
    float *grad_values_d;
    cudaMalloc((void**)&grad_values_d, out_size * sizeof(float));
    cudaMemset(grad_values_d, 0, out_size * sizeof(float));
    
    Tensor *dX = (Tensor *)malloc(sizeof(Tensor));
    dX->num_dim = tensor->num_dim;
    dX->dim = (uint32_t *)malloc(dX->num_dim * sizeof(uint32_t));
    memcpy(dX->dim, tensor->dim, dX->num_dim * sizeof(uint32_t));

    uint32_t grid_height = ceil(feature_height * 1.0 / TILE_WIDTH);
    uint32_t grid_width = ceil(feature_width * 1.0 / TILE_WIDTH);
    uint32_t out_tiles_num = grid_width * grid_height;

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(num_channels, out_tiles_num, num_samples);

    SigmoidForwardKernel<<<dimGrid, dimBlock>>>(
        tensor->values_d, Y_d,
        grad_values_d,
        grid_height, grid_width,
        feature_height, feature_width
    );

    dX->values_d = grad_values_d;
    grad->dW_or_W = NULL;
    grad->dX_or_X = dX;
    grad->is_grad = true;

    // Update tensor.
    cudaFree(tensor->values_d);
    tensor->values_d = Y_d;
}


void run_sigmoid_backward(LayerGradients *grad, LayerGradients *next_layer_grad) {
    Tensor *dX = grad->dX_or_X;
    uint32_t num_samples    = dX->dim[0];
    uint32_t num_channels   = dX->dim[1];
    uint32_t feature_height = dX->dim[2];
    uint32_t feature_width  = dX->dim[3];

    uint32_t grid_height = ceil(feature_height * 1.0 / TILE_WIDTH);
    uint32_t grid_width = ceil(feature_width * 1.0 / TILE_WIDTH);
    uint32_t out_tiles_num = grid_width * grid_height;

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(num_channels, out_tiles_num, num_samples);
    MultiplyKernel<<<dimGrid, dimBlock>>>(
        dX->values_d, next_layer_grad->dX_or_X->values_d,
        grid_height, grid_width,
        feature_height, feature_width
    );
}


// Assume stride is always the kernel size.
void run_pooling_forward(Tensor *tensor, uint32_t kernel_length, pooling_type pool_type, LayerGradients *grad) {
    uint32_t num_samples    = tensor->dim[0];
    uint32_t num_channels   = tensor->dim[1];
    uint32_t feature_height = tensor->dim[2];
    uint32_t feature_width  = tensor->dim[3];
    uint32_t out_height     = feature_height / kernel_length;
    uint32_t out_width      = feature_width / kernel_length;
    uint32_t in_size        = num_samples * num_channels * feature_height * feature_width;
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

    // Store gradients.
    float *grad_values_d;
    cudaMalloc((void**)&grad_values_d, in_size * sizeof(float));
    cudaMemset(grad_values_d, 0, in_size * sizeof(float));
    
    Tensor *dX = (Tensor *)malloc(sizeof(Tensor));
    dX->num_dim = tensor->num_dim;
    dX->dim = (uint32_t *)malloc(dX->num_dim * sizeof(uint32_t));
    memcpy(dX->dim, tensor->dim, dX->num_dim * sizeof(uint32_t));

    PoolForwardKernel<<<dimGrid, dimBlock>>>(
        tensor->values_d, Y_d,
        pool_type,
        grad_values_d,
        kernel_length,
        grid_height, grid_width,
        feature_height, feature_width,
        out_height, out_width
    );
    
    dX->values_d = grad_values_d;

    grad->dW_or_W = NULL;
    grad->dX_or_X = dX;
    grad->is_grad = true;

    // Update tensor.
    tensor->dim[2] = out_height;
    tensor->dim[3] = out_width;
    cudaFree(tensor->values_d);
    tensor->values_d = Y_d;
}


void run_pooling_backward(uint32_t kernel_length, LayerGradients *grad, LayerGradients *next_layer_grad) {
    Tensor *dX = grad->dX_or_X;
    uint32_t num_samples  = dX->dim[0];
    uint32_t num_channels = dX->dim[1];
    uint32_t grad_height  = dX->dim[2];
    uint32_t grad_width   = dX->dim[3];
    uint32_t next_layer_grad_height = grad_height / kernel_length;
    uint32_t next_layer_grad_width  =  grad_width / kernel_length;
    
    uint32_t grid_height = ceil(grad_height * 1.0 / TILE_WIDTH);
    uint32_t grid_width = ceil(grad_width * 1.0 / TILE_WIDTH);
    uint32_t grad_tiles_num = grid_width * grid_height;

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(num_channels, grad_tiles_num, num_samples);

    PoolBackwardKernel<<<dimGrid, dimBlock>>>(
        dX->values_d, next_layer_grad->dX_or_X->values_d,
        kernel_length,
        grid_height, grid_width,
        grad_height, grad_width,
        next_layer_grad_height, next_layer_grad_width
    );
}


void run_flatten_forward(Tensor *tensor) {
    // Make sure to keep the sample dimension.
    uint32_t num_samples = tensor->dim[0];
    uint32_t size = get_tensor_values_size(tensor->num_dim, tensor->dim) / num_samples;
    tensor->num_dim = 2;
    free(tensor->dim);

    tensor->dim = (uint32_t *)malloc(tensor->num_dim * sizeof(uint32_t));
    tensor->dim[0] = num_samples;
    tensor->dim[1] = size;
}


void run_flatten_backward(uint32_t num_samples, uint8_t kernel_length, LayerGradients *grad, LayerGradients *next_layer_grad) {
    grad->dW_or_W = NULL;
    
    // Update dX.
    Tensor *dX = deep_copy_tensor(next_layer_grad->dX_or_X);
    // Derive dimensions from num_samples and kernel_length:
    // out_size = num_samples * num_channels * kernel_length**2.
    uint32_t out_size     = get_tensor_values_size(dX->num_dim, dX->dim);
    uint32_t num_channels = out_size / (num_samples * kernel_length * kernel_length);
    dX->num_dim = 4;

    free(dX->dim);
    dX->dim = (uint32_t *)malloc(dX->num_dim * sizeof(uint32_t));
    dX->dim[0] = num_samples;
    dX->dim[1] = num_channels;
    dX->dim[2] = kernel_length;
    dX->dim[3] = kernel_length;

    grad->dX_or_X = dX;
    grad->is_grad = true;
}


void run_linear_forward(Tensor *X, Tensor *linear_weights, LayerGradients *grad) {
    uint32_t in_features  = linear_weights->dim[1];
    uint32_t out_features = linear_weights->dim[0];
    uint32_t num_samples  = X->dim[0];
    uint32_t out_size     = num_samples * out_features;

    // Store gradients.
    Tensor *dW = (Tensor *)malloc(sizeof(Tensor));
    Tensor *dX = (Tensor *)malloc(sizeof(Tensor));
    
    float *dW_values, *dX_values;
    cudaMalloc((void**)&dW_values, num_samples * in_features * sizeof(float));
    cudaMemcpy(dW_values, X->values_d, num_samples * in_features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&dX_values, out_features * in_features * sizeof(float));
    cudaMemcpy(dX_values, linear_weights->values_d, out_features * in_features * sizeof(float), cudaMemcpyHostToDevice);

    dW->dim = (uint32_t *)malloc(2 * sizeof(uint32_t));
    dW->dim[0] = num_samples;
    dW->dim[1] = in_features;
    dW->num_dim = 2;
    dW->values_d = dW_values;

    dX->dim = (uint32_t *)malloc(2 * sizeof(uint32_t));
    dX->dim[0] = out_features;
    dX->dim[1] = in_features;
    dX->num_dim = 2;
    dX->values_d = dX_values;

    grad->dW_or_W = dW;
    grad->dX_or_X = dX;
    grad->is_grad = true;

    // Run linear layer.
    float *Y_d;
    cudaMalloc((void**)&Y_d, out_size * sizeof(float));

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(ceil(out_features * 1.0 / (TILE_WIDTH * THREAD_COARSENING_FACTOR)), ceil(num_samples * 1.0 / TILE_WIDTH));
    LinearForwardKernel<<<dimGrid, dimBlock>>>(
        X->values_d,
        linear_weights->values_d,
        Y_d,
        num_samples,
        in_features, out_features
    );

    // Update tensor.
    X->dim[1] = out_features;
    cudaFree(X->values_d);
    X->values_d = Y_d;
}


void run_linear_backward(Tensor *linear_weights, LayerGradients *grad, LayerGradients *next_layer_grad, float lr) {
    // Recall dY has dimension [num_samples x out_features].
    Tensor *dY = next_layer_grad->dX_or_X;
    Tensor *dW = grad->dW_or_W;
    Tensor *dX = grad->dX_or_X;

    uint32_t num_samples  = dY->dim[0];
    uint32_t out_features = dY->dim[1];
    uint32_t in_features  = dW->dim[1];

    float *dYT, *updated_dW_d, *updated_dX_d;
    cudaMalloc((void**)&dYT, num_samples * out_features * sizeof(float));
    cudaMemset(dYT, 0, num_samples * out_features * sizeof(float));
    
    // Update dW = dY.T @ dW.
    // Transpose dY.
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGridTranspose(ceil(num_samples * 1.0 / TILE_WIDTH), ceil(out_features * 1.0 / TILE_WIDTH));
    TransposeMatrixKernel<<<dimGridTranspose, dimBlock>>>(dY->values_d, dYT, out_features, num_samples);

    cudaMalloc((void**)&updated_dW_d, out_features * in_features * sizeof(float));
    dim3 dimGridUpdateDW(ceil(in_features * 1.0 / TILE_WIDTH / THREAD_COARSENING_FACTOR), ceil(out_features * 1.0 / TILE_WIDTH));
    MatMulKernel<<<dimGridUpdateDW, dimBlock>>>(dYT, dW->values_d, updated_dW_d, out_features, num_samples, in_features);

    cudaFree(dW->values_d);
    dW->values_d = updated_dW_d;
    dW->dim[0]   = out_features;

    // Update dX = dY @ dX.
    dim3 dimGridUpdateDX(ceil(in_features * 1.0 / TILE_WIDTH / THREAD_COARSENING_FACTOR), ceil(num_samples * 1.0 / TILE_WIDTH));
    cudaMalloc((void**)&updated_dX_d, num_samples * in_features * sizeof(float));
    MatMulKernel<<<dimGridUpdateDX, dimBlock>>>(dY->values_d, dX->values_d, updated_dX_d, num_samples, out_features, in_features);
    
    cudaFree(dX->values_d);
    dX->values_d = updated_dX_d;
    dX->dim[0]   = num_samples;

    // Update weights W = W - lr * dW.
    dim3 dimGridUpdateW(ceil(in_features * 1.0 / TILE_WIDTH), ceil(out_features * 1.0 / TILE_WIDTH));
    Update2DGridParameterKernel<<<dimGridUpdateW, dimBlock>>>(linear_weights->values_d, updated_dW_d, out_features, in_features, lr);
}


/**
 * Perform softmax function on a 2D tensor across column.
 * TODO: enable performing the softmax on n-dimensional tensor given the input axis.
 * 
 */
void run_softmax_forward(Tensor *tensor, uint8_t *y_d, LayerGradients *grad) {
    if (tensor->num_dim != 2) {
        printf("The input tensor must have 2 dimensions to perform softmax function.\n");
        free_tensor(tensor);
        tensor = NULL;
        return;
    }

    uint32_t num_samples  = tensor->dim[0];
    uint32_t num_features = tensor->dim[1];
    uint32_t out_size     = num_samples * num_features;

    float *X_output_d, *X_exp_sum_d;
    cudaMalloc((void**)&X_output_d, out_size * sizeof(float));
    cudaMalloc((void**)&X_exp_sum_d, num_samples * sizeof(float));
    cudaMemset(X_exp_sum_d, 0, num_samples * sizeof(float));

    // TODO: consider the cases where the total size is significantly lower
    // than TILE_WIDTH * TILE_WIDTH;
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(ceil(num_features * 1.0 / TILE_WIDTH), ceil(num_samples * 1.0 / TILE_WIDTH));
    CalcExpAndSumByRowKernel<<<dimGrid, dimBlock>>>(
        tensor->values_d, X_output_d, X_exp_sum_d, num_samples, num_features
    );

    NormalizeKernel<<<dimGrid, dimBlock>>>(X_output_d, X_exp_sum_d, num_samples, num_features);

    cudaFree(X_exp_sum_d);
    
    // Update tensor.
    cudaFree(tensor->values_d);
    tensor->values_d = X_output_d;

    // Compute gradients assuming cross entropy loss.
    float *dX_d;
    cudaMalloc((void**)&dX_d, out_size * sizeof(float));
    cudaMemset(dX_d, 0, out_size * sizeof(float));
    SoftmaxGradientKernel<<<dimGrid, dimBlock>>>(dX_d, X_output_d, y_d, num_samples, num_features);

    Tensor *dX = (Tensor *)malloc(sizeof(Tensor));
    dX->num_dim = 2;
    dX->dim = (uint32_t *)malloc(2 * sizeof(uint32_t));
    dX->dim[0] = tensor->dim[0];
    dX->dim[1] = tensor->dim[1];
    dX->values_d = dX_d;

    grad->dW_or_W = NULL;
    grad->dX_or_X = dX;
    grad->is_grad = true;
}


Tensor *compute_negative_log_likelihood_log_lost(Tensor *tensor, uint8_t *y_d) {
    uint32_t num_samples  = tensor->dim[0];
    uint32_t num_features = tensor->dim[1];

    float *out;
    cudaMalloc((void**)&out, sizeof(float));
    cudaMemset(out, 0, sizeof(float));

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(ceil(LABEL_SIZE * 1.0 / (TILE_WIDTH * THREAD_COARSENING_FACTOR)), ceil(num_samples * 1.0 / TILE_WIDTH));
    NegativeLogLikelihoodLogKernel<<<dimGrid, dimBlock>>>(tensor->values_d, y_d, out, num_samples, num_features);

    Tensor *output = (Tensor *)malloc(sizeof(Tensor));
    output->num_dim = 1;
    output->dim = (uint32_t *)malloc(sizeof(uint32_t));
    output->dim[0] = 1;
    output->values_d = out;
    
    return output;
}
