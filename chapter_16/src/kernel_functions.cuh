#ifndef KERNELS
#define KERNELS


#include <stdint.h>

#define TILE_WIDTH 32
#define THREAD_COARSENING_FACTOR 8

enum pooling_type { MAX, MEAN };



__global__ void Conv2ForwardKernel(
    float *X, float *Y,
    float *filters,
    uint32_t kernel_length,
    uint32_t in_channels,
    uint32_t grid_height, uint32_t grid_width,
    uint32_t in_height, uint32_t in_width,
    uint32_t out_height, uint32_t out_width
);


__global__ void SigmoidForwardKernel(
    float *X, float *Y,
    uint32_t grid_height, uint32_t grid_width,
    uint32_t out_height, uint32_t out_width
);


__global__ void PoolForwardKernel(
    float *X, float *Y,
    pooling_type pool_type,
    uint32_t kernel_length,
    uint32_t grid_height, uint32_t grid_width,
    uint32_t in_height, uint32_t in_width,
    uint32_t out_height, uint32_t out_width
);


#endif