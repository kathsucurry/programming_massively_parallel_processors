#ifndef TEST_UTILS
#define TEST_UTILS


#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#include "../src/data_loader.cuh"
#include "../src/preprocessing.cuh"
#include "../src/common.cuh"
#include "../src/cnn_layers.cuh"


#define diff_eps 1e-5


void keep_first_n_samples(MNISTDataset *dataset, uint32_t n);
void print_sample(ImageDataset *dataset, uint32_t sample_index);
float *read_float_array_from_file(const char *file_path, uint32_t size);

void  compare_tensor(const char *label, const char *file_path, Tensor *tensor);
void  compare_float(const char *label, const char *file_path, float *value);

void print_flatten_tensor_values(Tensor *tensor, uint32_t print_size);

#endif