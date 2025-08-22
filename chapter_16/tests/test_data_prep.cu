#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#include "../src/data_loader.cuh"
#include "../src/preprocessing.cuh"
#include "../src/common.cuh"
#include "../src/cnn_layers.cuh"
#include "test_utils.cuh"
#include "test_data_prep.cuh"


int _compare_int_pointer(const void *a, const void *b) {
    return ( *(uint32_t*)a - *(uint32_t*)b );
}


void test_initialize_conv_layer_weights(Tensor *tensor) {
    printf("Data prep test: initialize conv2d layer weights.....");

    if (tensor->num_dim != 4) {
        printf("FAILED\n");
        printf("# dim must be 4; actual value = %u\n", tensor->num_dim);
        abort();
    }

    uint32_t expected_dim[] = {4, 1, 5, 5};
    for (uint32_t i = 0; i < tensor->num_dim; ++i)
        if (tensor->dim[i] != expected_dim[i]) {
            printf("FAILED\n");
            printf("Expected dim[%u] = %u; actual value = %u\n", i, expected_dim[i], tensor->dim[i]);
            abort();
        }

    float x = sqrtf(6.0 / (125));
    uint32_t out_size = 4 * 1 * 5 * 5;

    float values[out_size];
    cudaMemcpy(values, tensor->values_d, out_size * sizeof(float), cudaMemcpyDeviceToHost);
    for (uint32_t i = 0; i < out_size; ++i)
        if (values[i] <= -1 * x || values[i] >= x) {
            printf("FAILED\n");
            printf("Given x = %.3f, found value %.3f\n", x, values[i]);
            abort();
        }

    printf("PASSED\n");
}


void test_initialize_linear_layer_weights(Tensor *tensor) {
    printf("Data prep test: initialize linear layer weights.....");

    if (tensor->num_dim != 2) {
        printf("FAILED\n");
        printf("# dim must be 2; actual value = %u\n", tensor->num_dim);
        abort();
    }

    uint32_t expected_dim[] = {10, 784};
    for (uint32_t i = 0; i < tensor->num_dim; ++i)
        if (tensor->dim[i] != expected_dim[i]) {
            printf("FAILED\n");
            printf("Expected dim[%u] = %u; actual value = %u\n", i, expected_dim[i], tensor->dim[i]);
            abort();
        }

    float x = sqrtf(6.0 / (794));
    uint32_t out_size = 784 * 10;

    float values[out_size];
    cudaMemcpy(values, tensor->values_d, out_size * sizeof(float), cudaMemcpyDeviceToHost);
    for (uint32_t i = 0; i < out_size; ++i)
        if (values[i] <= -1 * x || values[i] >= x) {
            printf("FAILED\n");
            printf("Given x = %.3f, found value %.3f\n", x, values[i]);
            abort();
        }

    printf("PASSED\n");
}


void test_shuffle_indices(ImageDataset *dataset) {
    printf("Data prep test: shuffle indices.....");

    uint32_t num_samples = dataset->num_samples;
    uint32_t sorted_view_indices[num_samples];
    memcpy(sorted_view_indices, dataset->view_indices, sizeof(sorted_view_indices));
    std::qsort(sorted_view_indices, num_samples, sizeof(uint32_t), _compare_int_pointer);

    for (uint32_t i = 0; i < num_samples - 1; ++i)
        if (sorted_view_indices[i] + 1 != sorted_view_indices[i + 1]) {
            printf("FAILED\n");
            printf("Sorted view indices are not increasing by one: %u --> %u\n", sorted_view_indices[i], sorted_view_indices[i + 1]);
            abort();
        }
    
    printf("PASSED\n");
}


void test_prepare_batch(float X[], uint8_t y[], ImageDataset *dataset, bool include_visual_check) {
    printf("Data prep test: prepare batch (including visual check if set).....");
    uint32_t check_index  = 1;
    uint32_t image_height = 32;
    uint32_t image_width  = 32;
    
    uint8_t label = dataset->labels[dataset->view_indices[check_index]];

    if (include_visual_check) {
        // Print sample.
        printf("\n");
        for (uint32_t row = 0; row < image_height; ++row) {
            for (uint32_t col = 0; col < image_width; ++col)
                printf("%5.2f", X[check_index * image_height * image_width + row * image_width + col]);
            printf("\n");
        }
        printf("\nLabel: %u\n", label);
    }
    
    for (uint32_t i = 0; i < LABEL_SIZE; ++i) {
        uint8_t y_value = y[check_index * LABEL_SIZE + i]; 
        if ((i == label && y_value != 1) || (i != label && y_value != 0)) {
            printf("FAILED: one hot encoding y is not correct.\n");
            abort();
        }
    }

    if (include_visual_check)
        printf("[CONT] Data prep test: prepare batch.....");
    printf("PASSED\n");
}