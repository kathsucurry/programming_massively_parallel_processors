#ifndef TEST_DATA_PREP
#define TEST_DATA_PREP


#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#include "../src/data_loader.cuh"
#include "../src/preprocessing.cuh"
#include "../src/common.cuh"
#include "../src/cnn_layers.cuh"


void test_initialize_conv_layer_weights(Tensor *tensor);
void test_initialize_linear_layer_weights(Tensor *tensor);
void test_shuffle_indices(ImageDataset *dataset);
void test_prepare_batch(float X[], uint8_t y[], ImageDataset *dataset, bool include_visual_check);


#endif