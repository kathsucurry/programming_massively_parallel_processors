#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#include "../src/data_loader.cuh"
#include "../src/preprocessing.cuh"
#include "../src/common.h"
#include "../src/cnn_layers.cuh"
#include "test_utils.cuh"


void keep_first_n_samples(MNISTDataset *dataset, uint32_t n) {
    MNISTImage *images = (MNISTImage *)malloc(n * sizeof(MNISTImage));
    uint8_t *labels = (uint8_t *)malloc(n * sizeof(uint8_t));
    
    for (uint8_t i = 0; i < n; ++i) {
        MNISTImage image = dataset->images[i];
        uint8_t *pixels = (uint8_t *)malloc(image.height * image.width * sizeof(uint8_t));
        memcpy(pixels, image.pixels, image.height * image.width * sizeof(uint8_t));
        images[i].pixels = pixels;
        images[i].height = image.height;
        images[i].width  = image.width;
        labels[i] = dataset->labels[i];
    }

    // Update dataset.
    free_MNIST_images(dataset->images, dataset->num_samples);
    dataset->images = images;
    free(dataset->labels);
    dataset->labels = labels;
    dataset->num_samples = n;
}


void print_sample(ImageDataset *dataset, uint32_t sample_index) {
    printf("--> Printing sample with index %u...\n\n", sample_index);

    Image image   = dataset->images[sample_index];
    uint8_t label = dataset->labels[sample_index];

    printf("Label\t\t: %u\n", label);
    printf("Image size \t: %u %u\n", image.height, image.width);
    for (uint32_t row = 0; row < image.height; ++row) {
        for (uint32_t col = 0; col < image.width; ++col) {
            printf("%4.1f", image.pixels[row * image.width + col]);
        }
        printf("\n");
    }
    printf("\n");
}


float *read_float_array_from_file(const char *file_path, uint32_t size) {
    FILE *file;
    file = fopen(file_path, "r");
    
    float *x = (float *)malloc(size * sizeof(float));
    for (uint32_t i = 0; i < size; ++i)
        fscanf(file, "%f", &x[i]);

    fclose(file);
    return x;
}


void compare_results(const char *label, const char *file_path, Tensor *tensor) {
    uint32_t out_size = get_tensor_values_size(tensor->num_dim, tensor->dim);
    float *expected_output = read_float_array_from_file(file_path, out_size);

    printf("Test: %s...", label);

    float *actual_output = (float *)malloc(out_size * sizeof(float));
    cudaMemcpy(actual_output, tensor->values_d, out_size * sizeof(float), cudaMemcpyDeviceToHost);

    for (uint32_t i = 0; i < out_size; ++i)
        if (abs(actual_output[i] - expected_output[i]) > compare_eps) {
            printf("FAILED\n");
            abort();
        }
    printf("PASSED\n");

    free(actual_output);
}
