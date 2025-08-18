#ifndef KERNELS
#define KERNELS


#include <stdint.h>

#define TILE_WIDTH 2 // 32
#define THREAD_COARSENING_FACTOR 3 // 8
#define eps 1e-6

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
    float *X, float *Y, float *grad,
    uint32_t grid_height, uint32_t grid_width,
    uint32_t out_height, uint32_t out_width
);

__global__ void PoolForwardKernel(
    float *X, float *Y,
    pooling_type pool_type,
    float *grad,
    uint32_t kernel_length,
    uint32_t grid_height, uint32_t grid_width,
    uint32_t in_height, uint32_t in_width,
    uint32_t out_height, uint32_t out_width
);

__global__ void LinearForwardKernel(
    float *X, float *Y,
    float *linear_weights,
    uint32_t num_samples,
    uint32_t in_features, uint32_t out_features
);

__global__ void CalcExpAndSumByRowKernel(
    float *X, float *exp_X, float *sum_exp_X, uint32_t num_samples, uint32_t num_features
);

__global__ void NormalizeKernel(float *X, float *sum, uint32_t num_samples, uint32_t num_features);

__global__ void NegativeLogLikelihoodLogKernel(
    const float *X, const uint8_t *y, float *out, uint32_t num_samples, uint32_t num_features
);

__global__ void SoftmaxGradientKernel(
    float *dX_d, const float *output, const uint8_t *y, uint32_t num_samples, uint32_t num_features
);


#endif