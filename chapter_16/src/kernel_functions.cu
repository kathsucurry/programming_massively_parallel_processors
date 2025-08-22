#include <cuda_runtime.h>
#include <stdint.h>

#include "kernel_functions.cuh"
#include "common.cuh"


/**
 * (Not optimized) Conv2 kernel implementation, following the method in chapter 16.3 (Fig. 16.13,14).
 */
__global__ void Conv2DForwardKernel(
    float *X, float *Y,
    float *filters,
    uint32_t kernel_length,
    uint32_t in_channels,
    uint32_t grid_height, uint32_t grid_width,
    uint32_t in_height, uint32_t in_width
) {
    uint32_t out_height = in_height - kernel_length + 1;
    uint32_t out_width  = in_width - kernel_length + 1;
    uint32_t out_channel_idx = blockIdx.x;
    uint32_t out_height_idx  = (blockIdx.y / grid_width)*TILE_WIDTH + threadIdx.y;
    uint32_t out_width_idx   = (blockIdx.y % grid_width)*TILE_WIDTH + threadIdx.x;
    uint32_t sample_idx      = blockIdx.z;
    uint32_t out_channels = gridDim.x;

    if (out_height_idx >= out_height || out_width_idx >= out_width)
        return;
    
    float value = 0.0f;
    for (uint32_t in_channel_idx = 0; in_channel_idx < in_channels; ++in_channel_idx)
        for (uint32_t k_row = 0; k_row < kernel_length; ++k_row)
            for (uint32_t k_col = 0; k_col < kernel_length; ++k_col) {
                uint32_t in_row = out_height_idx + k_row;
                uint32_t in_col = out_width_idx + k_col;
                
                uint32_t X_idx = (sample_idx * in_channels * in_height * in_width) + 
                    (in_channel_idx * in_height * in_width) + 
                    (in_row * in_width) + 
                    in_col;
                uint32_t weight_idx = (out_channel_idx * in_channels * kernel_length * kernel_length) +
                    (in_channel_idx * kernel_length * kernel_length) +
                    (k_row * kernel_length) +
                    k_col;
                value += X[X_idx] * filters[weight_idx];
            }
    
    uint32_t Y_idx = (sample_idx * out_channels * out_height * out_width) + 
            (out_channel_idx * out_height * out_width) +
            (out_height_idx * out_width) +
            out_width_idx;
    Y[Y_idx] = value;
}


__global__ void Conv2DBackwardXGradKernel(
    float *dX, float *dY, float *W,
    uint32_t kernel_length,
    uint32_t out_channels,
    uint32_t grid_heigth, uint32_t grid_width,
    uint32_t in_height, uint32_t in_width
) {
    uint32_t out_height     = in_height - kernel_length + 1;
    uint32_t out_width      = in_width - kernel_length + 1;
    uint32_t in_channel_idx = blockIdx.x;
    uint32_t in_height_idx  = (blockIdx.y / grid_width)*TILE_WIDTH + threadIdx.y;
    uint32_t in_width_idx   = (blockIdx.y % grid_width)*TILE_WIDTH + threadIdx.x;
    uint32_t sample_idx     = blockIdx.z;
    uint32_t in_channels    = gridDim.x;

    if (in_height_idx >= in_height || in_width_idx >= in_width)
        return;
    
    float value = 0.0f;
    for (uint32_t out_channel_idx = 0; out_channel_idx < out_channels; ++out_channel_idx)
        for (uint32_t k_row = 0; k_row < kernel_length; ++k_row)
            for (uint32_t k_col = 0; k_col < kernel_length; ++k_col) {
                if (in_height_idx >= k_row && in_width_idx >= k_col &&
                        in_height_idx < out_height + k_row && in_width_idx < out_width + k_col) {
                    uint32_t out_row = in_height_idx - k_row;
                    uint32_t out_col = in_width_idx - k_col;
                
                    uint32_t dY_idx = (sample_idx * out_channels * out_height * out_width) + 
                        (out_channel_idx * out_height * out_width) + 
                        (out_row * out_width) + 
                        out_col;
                    uint32_t weight_idx = (out_channel_idx * in_channels * kernel_length * kernel_length) +
                        (in_channel_idx * kernel_length * kernel_length) +
                        ((kernel_length - k_row) * kernel_length) +
                        kernel_length - k_col;
                    value += dY[dY_idx] * W[weight_idx];
                }
            }
    
    uint32_t dX_idx = (sample_idx * in_channels * in_height * in_width) + 
            (in_channel_idx * in_height * in_width) +
            (in_height_idx * in_width) +
            in_width_idx;
    dX[dX_idx] = value;
}


__global__ void Conv2DBackwardWGradKernel(
    float *dW, float *dY, float *X,
    uint32_t kernel_length,
    uint32_t num_samples,
    uint32_t grid_height, uint32_t grid_width,
    uint32_t out_height, uint32_t out_width
) {
    uint32_t in_channel_idx  = blockIdx.x;
    uint32_t filter_row_idx  = (blockIdx.y / grid_width)*TILE_WIDTH + threadIdx.y;
    uint32_t filter_col_idx  = (blockIdx.y % grid_width)*TILE_WIDTH + threadIdx.x;
    uint32_t out_channel_idx = blockIdx.z;
    uint32_t in_channels     = gridDim.x;
    uint32_t out_channels    = gridDim.z;
    uint32_t in_height       = out_height + kernel_length - 1;
    uint32_t in_width        = out_width + kernel_length - 1;

    if (filter_row_idx >= kernel_length || filter_col_idx >= kernel_length)
        return;
    
    float value = 0.0f;
    for (uint32_t sample_idx = 0; sample_idx < num_samples; ++sample_idx)
        for (uint32_t out_row = 0; out_row < out_height; ++out_row)
            for (uint32_t out_col = 0; out_col < out_width; ++out_col) {
                uint32_t X_row = out_row + filter_row_idx;
                uint32_t X_col = out_col + filter_col_idx;
                
                uint32_t X_idx = (sample_idx * in_channels * in_height * in_width) + 
                    (in_channel_idx * in_height * in_width) + 
                    (X_row * in_width) + 
                    X_col;
                uint32_t dY_idx = (sample_idx * out_channels * out_height * out_width) + 
                    (out_channel_idx * out_height * out_width) + 
                    (out_row * out_width) + 
                    out_col;
                value += X[X_idx] * dY[dY_idx];
            }
    
    uint32_t dW_idx = (out_channel_idx * in_channels * kernel_length * kernel_length) +
            (in_channel_idx * kernel_length * kernel_length) +
            (filter_row_idx * kernel_length) +
            filter_col_idx;
    dW[dW_idx] = value;
}


__global__ void SigmoidForwardKernel(
    float *X, float *Y,
    float *grad,
    uint32_t grid_height, uint32_t grid_width,
    uint32_t out_height, uint32_t out_width
) {
    uint32_t out_channel_idx = blockIdx.x;
    uint32_t out_height_idx  = (blockIdx.y / grid_width)*TILE_WIDTH + threadIdx.y;
    uint32_t out_width_idx   = (blockIdx.y % grid_width)*TILE_WIDTH + threadIdx.x;
    uint32_t sample_idx      = blockIdx.z;
    uint32_t num_channels = gridDim.x;

    if (out_height_idx >= out_height || out_width_idx >= out_width)
        return;

    for (uint32_t row = 0; row < out_height; ++row)
        for (uint32_t col = 0; col < out_width; ++col) {
            uint32_t index = (sample_idx * num_channels * out_height * out_width) +
                (out_channel_idx * out_height * out_width) +
                (row * out_width) +
                col;
            float value = 1.0 / (1 + expf(-1 * X[index]));
            Y[index] = value;
            grad[index] = value * (1 - value);
        }
}


__global__ void MultiplyKernel(
    float *X1, float *X2,
    uint32_t grid_height, uint32_t grid_width,
    uint32_t feature_height, uint32_t feature_width
) {
    uint32_t num_channel_idx = blockIdx.x;
    uint32_t out_height_idx  = (blockIdx.y / grid_width)*TILE_WIDTH + threadIdx.y;
    uint32_t out_width_idx   = (blockIdx.y % grid_width)*TILE_WIDTH + threadIdx.x;
    uint32_t sample_idx      = blockIdx.z;
    uint32_t num_channels = gridDim.x;

    if (out_height_idx >= feature_height || out_width_idx >= feature_width)
        return;
    
    uint32_t index = (sample_idx * num_channels * feature_height * feature_width) + 
            (num_channel_idx * feature_height * feature_width) +
            (out_height_idx * feature_width) +
            out_width_idx;
    X1[index] *= X2[index];
}


/**
 * Perform either max or mean pooling forward layer.
 * 
 * For now, assume that the stride is always kernel_length and the input width & height
 * are always divisible by kernel_length.
 */
__global__ void PoolForwardKernel(
    float *X, float *Y,
    pooling_type pool_type,
    float *grad,
    uint32_t kernel_length,
    uint32_t grid_height, uint32_t grid_width,
    uint32_t in_height, uint32_t in_width,
    uint32_t out_height, uint32_t out_width
) {
    uint32_t num_channel_idx = blockIdx.x;
    uint32_t out_height_idx  = (blockIdx.y / grid_width)*TILE_WIDTH + threadIdx.y;
    uint32_t out_width_idx   = (blockIdx.y % grid_width)*TILE_WIDTH + threadIdx.x;
    uint32_t sample_idx      = blockIdx.z;
    uint32_t num_channels = gridDim.x;

    if (out_height_idx >= out_height || out_width_idx >= out_width)
        return;
    
    float value = 0.0f;
    uint32_t max_idx = 0;
    for (uint32_t k_row = 0; k_row < kernel_length; ++k_row)
        for (uint32_t k_col = 0; k_col < kernel_length; ++k_col) {
            uint32_t in_row = kernel_length * out_height_idx + k_row;
            uint32_t in_col = kernel_length * out_width_idx + k_col;
            
            uint32_t X_idx = (sample_idx * num_channels * in_height * in_width) + 
                (num_channel_idx * in_height * in_width) + 
                (in_row * in_width) + 
                in_col;
            
            if (pool_type == MAX) {
                if (X[X_idx] > value) {
                    max_idx = X_idx;
                    value = X[X_idx];
                }
            } else {
                value += (X[X_idx] / (kernel_length * kernel_length));
                grad[X_idx] = 1.0 / (kernel_length * kernel_length);
            }
        }
    
    if (pool_type == MAX)
        grad[max_idx] = 1.0;

    uint32_t Y_idx = (sample_idx * num_channels * out_height * out_width) + 
            (num_channel_idx * out_height * out_width) +
            (out_height_idx * out_width) +
            out_width_idx;
    Y[Y_idx] = value;
}


__global__ void PoolBackwardKernel(
    float *grad, float *next_layer_grad,
    uint32_t kernel_length,
    uint32_t grid_height, uint32_t grid_width,
    uint32_t grad_height, uint32_t grad_width,
    uint32_t next_layer_grad_height, uint32_t next_layer_grad_width
) {
    uint32_t num_channel_idx = blockIdx.x;
    uint32_t grad_height_idx = (blockIdx.y / grid_width)*TILE_WIDTH + threadIdx.y;
    uint32_t grad_width_idx  = (blockIdx.y % grid_width)*TILE_WIDTH + threadIdx.x;
    uint32_t sample_idx      = blockIdx.z;
    uint32_t num_channels = gridDim.x;

    if (grad_height_idx >= grad_height || grad_width_idx >= grad_width)
        return;
    
    uint32_t index = (sample_idx * num_channels * grad_height * grad_width) + 
            (num_channel_idx * grad_height * grad_width) +
            (grad_height_idx * grad_width) +
            grad_width_idx;
    uint32_t next_layer_grad_index = (sample_idx * num_channels * next_layer_grad_height * next_layer_grad_width) + 
            (num_channel_idx * next_layer_grad_height * next_layer_grad_width) +
            (grad_height_idx / kernel_length * next_layer_grad_width) +
            grad_width_idx / kernel_length;
    grad[index] *= next_layer_grad[next_layer_grad_index];
}


__global__ void LinearForwardKernel(
    float *X, float *A, float *Y,
    uint32_t num_samples,
    uint32_t in_features, uint32_t out_features
) {
    __shared__ float  shared_X[TILE_WIDTH][TILE_WIDTH];
    __shared__ float shared_AT[TILE_WIDTH][TILE_WIDTH];

    // Identify the row and column of the output element.
    uint32_t row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    uint32_t col_offset = blockIdx.x * TILE_WIDTH * THREAD_COARSENING_FACTOR + threadIdx.x;

    // Initialize values.
    float out_values[THREAD_COARSENING_FACTOR];
    for (uint8_t c = 0; c < THREAD_COARSENING_FACTOR; ++c)
        out_values[c] = 0.0f;
    
    // Loop over tiles.
    for (uint32_t phase = 0; phase < ceil(in_features * 1.0 / TILE_WIDTH); ++phase) {
        uint32_t X_index = row * in_features + phase * TILE_WIDTH + threadIdx.x;

        // Collaboratively load the features1 tile into shared memory.
        if (row < num_samples && phase * TILE_WIDTH + threadIdx.x < in_features)
            shared_X[threadIdx.y][threadIdx.x] = X[X_index];
        else
            shared_X[threadIdx.y][threadIdx.x] = 0.0f;
    
        for (uint8_t c = 0; c < THREAD_COARSENING_FACTOR; ++c) {
            uint32_t col = col_offset + c * TILE_WIDTH;
            uint32_t A_index = col * in_features + phase * TILE_WIDTH + threadIdx.y;

            // Collaboratively load the features2 tile into shared memory.
            if (phase * TILE_WIDTH + threadIdx.y < in_features && col < out_features)
                shared_AT[threadIdx.y][threadIdx.x] = A[A_index];
            else
                shared_AT[threadIdx.y][threadIdx.x] = 0.0f;
            __syncthreads();

            for (uint32_t i = 0; i < TILE_WIDTH; ++i)
                out_values[c] += shared_X[threadIdx.y][i] * shared_AT[i][threadIdx.x];
            __syncthreads();
        }
    }

    for (uint8_t c = 0; c < THREAD_COARSENING_FACTOR; ++c) {
        uint32_t col = col_offset + c * TILE_WIDTH;
        if (row < num_samples && col < out_features)
            Y[row * out_features + col] = out_values[c];
    }
}


__global__ void MatMulKernel(   
    float *X, float *A,
    float *Y,
    uint32_t X_height, uint32_t X_width, uint32_t A_width
) {
    __shared__ float  shared_X[TILE_WIDTH][TILE_WIDTH];
    __shared__ float  shared_A[TILE_WIDTH][TILE_WIDTH];

    // Identify the row and column of the output element.
    uint32_t row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    uint32_t col_offset = blockIdx.x * TILE_WIDTH * THREAD_COARSENING_FACTOR + threadIdx.x;

    // Initialize values.
    float out_values[THREAD_COARSENING_FACTOR];
    for (uint8_t c = 0; c < THREAD_COARSENING_FACTOR; ++c)
        out_values[c] = 0.0f;
    
    // Loop over tiles.
    for (uint32_t phase = 0; phase < ceil(X_width * 1.0 / TILE_WIDTH); ++phase) {
        uint32_t X_index = row * X_width + phase * TILE_WIDTH + threadIdx.x;

        // Collaboratively load the features1 tile into shared memory.
        if (row < X_height && phase * TILE_WIDTH + threadIdx.x < X_width)
            shared_X[threadIdx.y][threadIdx.x] = X[X_index];
        else
            shared_X[threadIdx.y][threadIdx.x] = 0.0f;
    
        for (uint8_t c = 0; c < THREAD_COARSENING_FACTOR; ++c) {
            uint32_t col = col_offset + c * TILE_WIDTH;
            uint32_t A_index = (phase * TILE_WIDTH + threadIdx.y) * A_width + col;

            // Collaboratively load the features2 tile into shared memory.
            if (phase * TILE_WIDTH + threadIdx.y < X_width && col < A_width)
                shared_A[threadIdx.y][threadIdx.x] = A[A_index];
            else
                shared_A[threadIdx.y][threadIdx.x] = 0.0f;
            __syncthreads();

            for (uint32_t i = 0; i < TILE_WIDTH; ++i)
                out_values[c] += shared_X[threadIdx.y][i] * shared_A[i][threadIdx.x];
            __syncthreads();
        }
    }

    for (uint8_t c = 0; c < THREAD_COARSENING_FACTOR; ++c) {
        uint32_t col = col_offset + c * TILE_WIDTH;
        if (row < X_height && col < A_width)
            Y[row * A_width + col] = out_values[c];
    }
}


__global__ void TransposeMatrixKernel(float *X, float *Y, uint32_t Y_height, uint32_t Y_width) {
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
   
    if (col >= Y_width || row >= Y_height)
        return;

    Y[row * Y_width + col] = X[col * Y_height + row];
}


__global__ void Update2DGridParameterKernel(float *W, float *dW, uint32_t height, uint32_t width, float lr) {
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col >= width || row >= height)
        return;

    W[row * width + col] -= lr * dW[row * width + col];
}


__global__ void Update3DGridParameterKernel(
    float *W, float *dW, uint32_t height, uint32_t width, uint32_t grid_height, uint32_t grid_width, float lr
) {
    uint32_t in_channel_idx  = blockIdx.x;
    uint32_t in_channels     = gridDim.x;
    uint32_t filter_row_idx  = (blockIdx.y / grid_width)*TILE_WIDTH + threadIdx.y;
    uint32_t filter_col_idx  = (blockIdx.y % grid_width)*TILE_WIDTH + threadIdx.x;
    uint32_t out_channel_idx = blockIdx.z;
    
    if (filter_row_idx >= height || filter_col_idx >= width)
        return;
    
    uint32_t index = (out_channel_idx * in_channels * height * width) +
        (in_channel_idx * height * width) +
        (filter_row_idx * width) + filter_col_idx;

    W[index] -= lr * dW[index];
}


__global__ void CalcExpAndSumByRowKernel(
    float *X, float *exp_X, float *sum_exp_X, uint32_t num_samples, uint32_t num_features
) {
    uint32_t col = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t row = blockDim.y * blockIdx.y + threadIdx.y;
    if (col >= num_features || row >= num_samples)
        return;
    
    float value = expf(X[row * num_features + col]);
    atomicAdd(&sum_exp_X[row], value);
    exp_X[row * num_features + col] = value;
}


__global__ void NormalizeKernel(float *X, float *sum, uint32_t num_samples, uint32_t num_features) {
    uint32_t col = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t row = blockDim.y * blockIdx.y + threadIdx.y;
    if (col >= num_features || row >= num_samples)
        return;
    
    X[row * num_features + col] /= sum[row];
}


__global__ void NegativeLogLikelihoodLogKernel(const float *X, const uint8_t *y, float *out, uint32_t num_samples, uint32_t num_features) {
    uint32_t col_offset = blockDim.x * blockIdx.x * THREAD_COARSENING_FACTOR + threadIdx.x;
    uint32_t row        = blockDim.y * blockIdx.y + threadIdx.y;

    if (row >= num_samples)
        return;

    float sum_value = 0;
    
    for (uint8_t c = 0; c < THREAD_COARSENING_FACTOR; ++c) {
        uint32_t col = col_offset + c * TILE_WIDTH;
        if (col >= num_features)
            break;

        sum_value += (-1 * logf(X[row * num_features + col] + 1e-6) * y[row * num_features + col]) / num_samples;
    }

    atomicAdd(out, sum_value);
}


__global__ void SoftmaxGradientKernel(float *dX_d, const float *output, const uint8_t *y, uint32_t num_samples, uint32_t num_features) {
    uint32_t col = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t row = blockDim.y * blockIdx.y + threadIdx.y;
    if (col >= num_features || row >= num_samples)
        return;

    // Make sure to normalize by the number of samples.
    dX_d[row * num_features + col] = (output[row * num_features + col] - y[row * num_features + col]) / num_samples;
}
