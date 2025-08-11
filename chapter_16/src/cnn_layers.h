#ifndef CNN_LAYERS
#define CNN_LAYERS


#include <stdint.h>


typedef struct {
    uint8_t num_filters;
    float **filters;
    uint8_t filter_width;
} Conv2DLayerWeights;

typedef struct {
    uint8_t matrix_height;
    uint8_t matrix_width;
    float *matrix;
} LinearLayerWeights;

typedef struct {
    Conv2DLayerWeights* conv2_weights;
    LinearLayerWeights* linear_weights;
} CNNNetworkWeights;


uint32_t *shuffle_indices(uint32_t num_samples, uint8_t seed);
float *xavier_initialization() {}


#endif