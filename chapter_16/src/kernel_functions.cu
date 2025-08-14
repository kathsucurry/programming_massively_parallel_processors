#include <cuda_runtime.h>
#include <stdint.h>

#include "kernel_functions.cuh"


/**
 * (Not optimized) Conv2 kernel implementation, following the method in chapter 16.3 (Fig. 16.13,14).
 */
__global__ void Conv2ForwardKernel(
    float *X, float *Y,
    float *filters,
    uint32_t kernel_length,
    uint32_t in_channels,
    uint32_t grid_height, uint32_t grid_width,
    uint32_t in_height, uint32_t in_width,
    uint32_t out_height, uint32_t out_width
) {
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


__global__ void SigmoidForwardKernel(
    float *X, float *Y,
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
            Y[index] = 1.0 / (1 + expf(-1 * X[index]));
        }
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
    for (uint32_t k_row = 0; k_row < kernel_length; ++k_row)
        for (uint32_t k_col = 0; k_col < kernel_length; ++k_col) {
            uint32_t in_row = kernel_length * out_height_idx + k_row;
            uint32_t in_col = kernel_length * out_width_idx + k_col;
            
            uint32_t X_idx = (sample_idx * num_channels * in_height * in_width) + 
                (num_channel_idx * in_height * in_width) + 
                (in_row * in_width) + 
                in_col;
            
            if (pool_type == MAX) {
                value = max(value, X[X_idx]);
            } else {
                value += (X[X_idx] / (kernel_length * kernel_length));
            }
        }

    uint32_t Y_idx = (sample_idx * num_channels * out_height * out_width) + 
            (num_channel_idx * out_height * out_width) +
            (out_height_idx * out_width) +
            out_width_idx;
    Y[Y_idx] = value;
}
