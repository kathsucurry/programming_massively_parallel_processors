/**
 * Perform mini test for the training. TODO: write unit tests.
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#include "src/data_loader.cuh"
#include "src/preprocessing.cuh"
#include "src/common.h"
#include "src/cnn_layers.cuh"


void print_conv2d_weight_init_example(uint32_t in_channels, uint32_t out_channels, uint8_t filter_size) {
    printf("--> Assess the conv layer weight initialization...");
    
    Tensor *conv2d_weight = initialize_conv_layer_weights(1, 3, 3, 3);
    printf("Conv2 weight has %u dimensions: ", conv2d_weight->num_dim);

    for (uint8_t i = 0; i < conv2d_weight->num_dim; ++i)
        printf("%u ", conv2d_weight->dim[i]);
    printf("\n");
    uint32_t weight_size = get_tensor_values_size(conv2d_weight->num_dim, conv2d_weight->dim);
    printf("Weight size: %u\n", weight_size);

    printf("Weight values: ");
    float *weights = (float *)malloc(weight_size * sizeof(float));
    cudaMemcpy(weights, conv2d_weight->values_d, weight_size * sizeof(float), cudaMemcpyDeviceToHost);

    for (uint32_t i = 0; i < weight_size; ++i)
        printf("%.2f ", weights[i]);
    printf("\n-------\n\n");

    free(weights);
    free_tensor(conv2d_weight);
}


/**
 * Generate a tiny dataset where all images are the same with float pixel values [0..image_height * image_width);
 */
ImageDataset *generate_tiny_dataset(uint32_t num_samples, uint32_t image_height, uint32_t image_width) {
    printf("--> Generate a tiny dataset with %u samples and images of [%u x %u]...\n", num_samples, image_height, image_width);

    ImageDataset *dataset = (ImageDataset *)malloc(sizeof(ImageDataset));
    Image *images = (Image *)malloc(num_samples * sizeof(Image));
    uint32_t *view_indices = (uint32_t *)malloc(num_samples * sizeof(uint32_t));
    for (uint8_t image_index = 0; image_index < num_samples; ++image_index) {
        view_indices[image_index] = image_index;
        float *pixels = (float *)malloc(image_height * image_width * sizeof(float));
        for (uint8_t i = 0; i < image_height * image_width; ++i)
            pixels[i] = i;
        images[image_index].height = image_height;
        images[image_index].width = image_width;
        images[image_index].pixels = pixels;
    }
    dataset->images = images;
    dataset->labels = NULL;
    dataset->num_samples = num_samples;
    dataset->view_indices = view_indices;

    printf("Image example:\n");
    Image image = dataset->images[0];
    for (uint32_t row = 0; row < image.width; ++row) {
        for (uint32_t col = 0; col < image.width; ++col) {
            uint32_t index = row * image.width + col;
            printf("%3.0f", image.pixels[index]);
        }
        printf("\n");
    }
    printf("\n");
    return dataset;
}


Tensor *generate_custom_weights(uint32_t in_channels, uint32_t out_channels, uint8_t kernel_length) {
    printf("--> Generate a custom weights...\n");
    
    Tensor *weights = (Tensor *)malloc(sizeof(Tensor));
    
    weights->num_dim = 4;
    uint32_t *dim = (uint32_t *)malloc(4 * sizeof(uint32_t));
    dim[0] = out_channels;
    dim[1] = in_channels;
    dim[2] = kernel_length;
    dim[3] = kernel_length;
    weights->dim = dim;

    printf("Conv2 weight to be used for further testing has %u dimensions: ", weights->num_dim);
    for (uint8_t i = 0; i < weights->num_dim; ++i)
        printf("%u ", dim[i]);
    printf("\n");

    uint32_t weight_size = get_tensor_values_size(weights->num_dim, dim);

    float *values = (float *)malloc(weight_size * sizeof(float));
    for (uint32_t filter_index = 0; filter_index < out_channels; ++filter_index) {
        float counter = 0.0f;
        for (uint32_t row = 0; row < kernel_length; ++row)
            for (uint32_t col = 0; col < kernel_length; ++col) {
                uint32_t index = filter_index * kernel_length * kernel_length + row * kernel_length + col;
                values[index] = counter++;
            }
    }
    printf("Weight values: ");
    for (uint32_t i = 0; i < weight_size; ++i)
        printf("%.0f ", values[i]);
    printf("\n\n");

    float *values_d;
    cudaMalloc((void**)&values_d, weight_size * sizeof(float));
    cudaMemcpy(values_d, values, weight_size * sizeof(float), cudaMemcpyHostToDevice);
    weights->values_d = values_d;

    return weights;
}


float *generate_X_device(ImageDataset *dataset) {
    uint32_t image_height = dataset->images[0].height;
    uint32_t image_width = dataset->images[0].width;
    float *X = (float *)malloc(dataset->num_samples * image_height * image_width * sizeof(float));
    for (uint32_t sample_index = 0; sample_index < dataset->num_samples; ++sample_index) {
        uint32_t sample_offset = sample_index * image_height * image_width;
        Image image = dataset->images[sample_index];
        for (uint32_t row = 0; row < image_height; ++row)
            for (uint32_t col = 0; col < image_width; ++col) {
                X[sample_offset + row * image_width + col] = image.pixels[row * image_width + col];
            }
    }
    
    float *X_d;
    cudaMalloc((void**)&X_d, dataset->num_samples * image_height * image_width * sizeof(float));
    cudaMemcpy(X_d, X, dataset->num_samples * image_height * image_width * sizeof(float), cudaMemcpyHostToDevice);

    free(X);
    return X_d;
}


void run_conv2d_forward_test() {
    printf("--> Perform conv2d test...\n");

    // Create a small ImageDataset; each image is only 5 x 5.
    uint32_t num_samples = 3;
    uint32_t image_height = 5;
    uint32_t image_width = 5;
    ImageDataset *dataset = generate_tiny_dataset(num_samples, image_height, image_width);
    float *X_d = generate_X_device(dataset);

    // Generate custom weights.
    uint32_t kernel_length = 3;
    uint32_t num_kernels = 3;
    uint32_t in_channels = 1;
    uint32_t show_sample_index = 0;
    Tensor *conv2d_weight = generate_custom_weights(in_channels, num_kernels, kernel_length);

    Tensor *output = initialize_tensor();
    run_conv2d_forward(output, X_d, conv2d_weight, num_samples, image_height, image_width);
    
    printf("Output description:\n");
    printf("# Dim: %u [", output->num_dim);
    for (uint8_t i = 0; i < output->num_dim; ++i)
        printf("%2u", output->dim[i]);
    printf("]\n\n");

    printf("Printing sample index %u:\n\n", show_sample_index);
    uint32_t sample_size = get_tensor_values_size(output->num_dim, output->dim) / output->dim[0];

    float *values = (float *)malloc(sample_size * sizeof(float));
    cudaMemcpy(values, &output->values_d[show_sample_index*sample_size], sample_size * sizeof(float), cudaMemcpyDeviceToHost);

    uint32_t feature_map_size = output->dim[2] * output->dim[3];

    for (uint32_t feature_index = 0; feature_index < output->dim[1]; ++feature_index) {
        printf("Feature map %u:\n", feature_index);
        for (uint32_t row = 0; row < output->dim[2]; ++row) {
            for (uint32_t col = 0; col < output->dim[3]; ++col) {
                printf("%4.0f", values[feature_index * feature_map_size + row * output->dim[3] + col]);
            }
            printf("\n");
        }
    }
    printf("\n");

    free_tensor(output);
    free_tensor(conv2d_weight);
    cudaFree(X_d);
    free_image_dataset(dataset);
}


Tensor *generate_test_rectangle_tensor(uint32_t height, uint32_t width, float multiplier) {
    Tensor *tensor = (Tensor *)malloc(sizeof(Tensor));
    tensor->num_dim = 2;
    uint32_t *dim = (uint32_t *)malloc(2 * sizeof(uint32_t));
    dim[0] = height;
    dim[1] = width;
    tensor->dim = dim;
    
    float counter = 0.0f;
    float *values = (float *)malloc(height * width * sizeof(float));
    for (uint32_t row = 0; row < height; ++row) {
        for (uint32_t col = 0; col < width; ++col) {
            uint32_t index = row * width + col;
            values[index] = multiplier * counter++;
            printf("%8.3f", values[index]);
        }
        printf("\n");
    }
    printf("\n");
    
    float *values_d;
    cudaMalloc((void**)&values_d, height * width * sizeof(float));
    cudaMemcpy(values_d, values, height * width * sizeof(float), cudaMemcpyHostToDevice);
    tensor->values_d = values_d;

    free(values);
    return tensor;
}


void run_linear_layer_test() {
    printf("--> Perform linear layer test...\n");

    uint32_t feature1_height = 4;
    uint32_t feature1_width  = 5;
    uint32_t feature2_width  = 7;

    // Prepare feature 1.
    printf("Generating X:\n");
    Tensor *X = generate_test_rectangle_tensor(feature1_height, feature1_width, 1);
    printf("Generating A:\n");
    Tensor *A = generate_test_rectangle_tensor(feature2_width, feature1_width, 1);

    // Recall that the output will be stored in feature1.
    run_linear_forward(X, A);

    uint32_t out_height = feature1_height;
    uint32_t out_width  = feature2_width;
    float *values = (float *)malloc(out_height * out_width * sizeof(float));
    cudaMemcpy(values, X->values_d, out_height * out_width * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Output: \n");
    for (uint32_t row = 0; row < out_height; ++row) {
        for (uint32_t col = 0; col < out_width; ++col) {
            printf("%8.0f", values[row * out_width + col]);
        }
        printf("\n");
    }
    printf("\n");

    free(values);

    free_tensor(X);
    free_tensor(A);
}


void run_log_softmax_forward_test() {
    printf("--> Perform softmax layer test...\n");
    printf("Input:\n");
    uint32_t num_samples  = 5;
    uint32_t num_features = 10;
    Tensor *X = generate_test_rectangle_tensor(num_samples, num_features, 0.01);

    run_log_softmax_forward(X);

    float *values = (float *)malloc(num_samples * num_features * sizeof(float));
    cudaMemcpy(values, X->values_d, num_samples * num_features * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Output: \n");
    for (uint32_t row = 0; row < num_samples; ++row) {
        for (uint32_t col = 0; col < num_features; ++col) {
            printf("%8.3f", values[row * num_features + col]);
        }
        printf("\n");
    }
    printf("\n");

    printf("# Dim: %u [", X->num_dim);
    for (uint8_t i = 0; i < X->num_dim; ++i)
        printf("%3u", X->dim[i]);
    printf("]\n\n");
    
    free(values);
    free_tensor(X);
}


void run_negative_log_likelihood_loss_test() {
    printf("--> Perform nll loss test...\n");
    
    // Build the log softmax input.
    uint32_t num_samples  = 5;
    uint32_t num_labels = 10;
    Tensor *X = generate_test_rectangle_tensor(num_samples, num_labels, 0.01);

    run_log_softmax_forward(X);

    float *values = (float *)malloc(num_samples * num_labels * sizeof(float));
    cudaMemcpy(values, X->values_d, num_samples * num_labels * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Input (log softmax): \n");
    for (uint32_t row = 0; row < num_samples; ++row) {
        for (uint32_t col = 0; col < num_labels; ++col) {
            printf("%8.3f", values[row * num_labels + col]);
        }
        printf("\n");
    }
    printf("\n");
    free(values);
    
    uint8_t *y = (uint8_t *)calloc(num_samples * num_labels, sizeof(uint8_t));
    for (uint32_t i = 0; i < num_samples; ++i)
        y[i * num_labels + i] = 1;
    printf("Input (labels):\n");
    for (uint32_t i = 0; i < num_samples; ++i) {
        for (uint8_t label = 0; label < num_labels; ++label)
            printf("%2u", y[i * num_labels + label]);
        printf("\n");
    }
    printf("\n");

    uint8_t *y_d;
    cudaMalloc((void**)&y_d, num_samples * num_labels * sizeof(uint8_t));
    cudaMemcpy(y_d, y, num_samples * num_labels * sizeof(uint8_t), cudaMemcpyHostToDevice);
    Tensor *loss = compute_negative_log_likelihood_lost(X, y_d);
    free_tensor(X);
    free(y);
    cudaFree(y_d);

    float *loss_h = (float *)malloc(sizeof(float));
    cudaMemcpy(loss_h, loss->values_d, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Output: %.3f\n\n", *loss_h);
    free(loss_h);
}


int main() {
    print_conv2d_weight_init_example(1, 3, 3);
    
    run_conv2d_forward_test();

    run_linear_layer_test();

    run_log_softmax_forward_test();

    run_negative_log_likelihood_loss_test();
}