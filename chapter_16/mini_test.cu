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


void print_tensor_2d(Tensor *tensor) {
    uint32_t out_size = get_tensor_values_size(tensor->num_dim, tensor->dim);
    float *values = (float *)malloc(out_size * sizeof(float));
    cudaMemcpy(values, tensor->values_d, out_size * sizeof(float), cudaMemcpyDeviceToHost);
    uint32_t *dim = tensor->dim;

    for (uint32_t row = 0; row < dim[0]; ++row) {
        for (uint32_t col = 0; col < dim[1]; ++col)
            printf("%10.2f", values[row * dim[1] + col]);
        printf("\n");
    }

    free(values);
}


void print_tensor_4d(Tensor *tensor) {
    uint32_t out_size = get_tensor_values_size(tensor->num_dim, tensor->dim);
    float *values = (float *)malloc(out_size * sizeof(float));
    cudaMemcpy(values, tensor->values_d, out_size * sizeof(float), cudaMemcpyDeviceToHost);
    uint32_t *dim = tensor->dim;

    for (uint32_t sample_index = 0; sample_index < dim[0]; ++sample_index) {
        printf("Sample index %u\n", sample_index);
        for (uint32_t channel_index = 0; channel_index < dim[1]; ++channel_index) {
            printf("Channel index %u\n", channel_index);
            for (uint32_t row = 0; row < dim[2]; ++row) {
                for (uint32_t col = 0; col < dim[3]; ++col)
                    printf("%6.2f", values[sample_index * out_size / dim[0] + channel_index * dim[2] * dim[3] + row * dim[3] + col]);
                printf("\n");
            }
            printf("\n");
        }
        printf("\n");
    }

    free(values);
}


Tensor *generate_test_ordered_tensor_2d(uint32_t height, uint32_t width, float multiplier) {
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


Tensor *generate_test_random_4d_tensor(uint32_t *dim, float multiplier, int seed) {
    srand(seed);

    Tensor *tensor = (Tensor *)malloc(sizeof(Tensor));
    tensor->num_dim = 4;
    
    uint32_t *tensor_dim = (uint32_t *)malloc(4 * sizeof(uint32_t));
    memcpy(tensor_dim, dim, 4 * sizeof(uint32_t));
    tensor->dim = tensor_dim;
    
    uint32_t out_size = get_tensor_values_size(4, dim);
    float *values = (float *)malloc(out_size * sizeof(float));
    for (uint32_t i = 0; i < out_size; ++i)
        values[i] = multiplier * (rand() % 50);
    
    float *values_d;
    cudaMalloc((void**)&values_d, out_size * sizeof(float));
    cudaMemcpy(values_d, values, out_size * sizeof(float), cudaMemcpyHostToDevice);
    tensor->values_d = values_d;
    free(values);
    return tensor;
}


void test_transpose_matrix() {
    printf("--> Test transpose matrix...\n");
    uint32_t input_height = 9;
    uint32_t input_width  = 7;
    printf("Input:\n");
    Tensor *input = generate_test_ordered_tensor_2d(input_height, input_width, 1);

    float *output_d;
    cudaMalloc((void**)&output_d, input_height * input_width * sizeof(float));
    cudaMemset(output_d, 0, input_height * input_width * sizeof(float));

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(ceil(input_height * 1.0 / TILE_WIDTH), ceil(input_width * 1.0 / TILE_WIDTH));
    TransposeMatrixKernel<<<dimGrid, dimBlock>>>(input->values_d, output_d, input_width, input_height);

    Tensor *output = (Tensor *)malloc(sizeof(Tensor));
    output->num_dim = 2;
    output->dim = (uint32_t *)malloc(2 * sizeof(uint32_t));
    output->dim[0] = input_width;
    output->dim[1] = input_height;
    output->values_d = output_d;

    printf("Output:\n");
    print_tensor_2d(output);
    printf("\n");

    free_tensor(input);
    free_tensor(output);
}


void test_matmul() {
    printf("--> Test matrix multiplication...\n");
    uint32_t input1_height = 9;
    uint32_t input1_width  = 7;
    uint32_t input2_width = 5;
    printf("Input 1:\n");
    Tensor *input1 = generate_test_ordered_tensor_2d(input1_height, input1_width, 1);

    printf("Input 2:\n");
    Tensor *input2 = generate_test_ordered_tensor_2d(input1_width, input2_width, 1);

    float *output_d;
    cudaMalloc((void**)&output_d, input1_height * input2_width * sizeof(float));
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(ceil(input2_width * 1.0 / TILE_WIDTH / THREAD_COARSENING_FACTOR), ceil(input1_height * 1.0 / TILE_WIDTH));
    MatMulKernel<<<dimGrid, dimBlock>>>(input1->values_d, input2->values_d, output_d, input1_height, input1_width, input2_width);

    Tensor *output = (Tensor *)malloc(sizeof(Tensor));
    output->num_dim = 2;
    output->dim = (uint32_t *)malloc(2 * sizeof(uint32_t));
    output->dim[0] = input1_height;
    output->dim[1] = input2_width;
    output->values_d = output_d;

    printf("Output:\n");
    print_tensor_2d(output);
    printf("\n");

    free_tensor(input1);
    free_tensor(input2);
    free_tensor(output);
}


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

    uint32_t *image_dim = (uint32_t *)malloc(3 * sizeof(uint32_t));
    image_dim[0] = num_samples;
    image_dim[1] = image_height;
    image_dim[2] = image_width;
    Tensor *output = initialize_tensor(X_d, 3, image_dim);
    LayerGradients *grad = (LayerGradients *)malloc(sizeof(LayerGradients));
    run_conv2d_forward(output, conv2d_weight, num_samples, image_height, image_width, grad);
    
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
    free_dataset(dataset);
}


void run_sigmoid_forward_test() {
    printf("--> Perform sigmoid layer test...\n");
    printf("Input:\n\n");

    uint8_t num_dim = 4;
    uint32_t *dim = (uint32_t *)malloc(num_dim * sizeof(uint32_t));
    dim[0] = 2; // # samples.
    dim[1] = 1; // # channels.
    dim[2] = 8; // Height.
    dim[3] = 6; // Width.
    Tensor *tensor = generate_test_random_4d_tensor(dim, 0.08, 0);
    print_tensor_4d(tensor);

    LayerGradients *grad = (LayerGradients *)malloc(sizeof(LayerGradients));

    run_sigmoid_forward(tensor, grad);

    printf("Sigmoid output:\n");
    print_tensor_4d(tensor);

    printf("Gradient:\n");
    print_tensor_4d(grad->dX_or_X);

    free_tensor(tensor);
    free_layer_gradients(grad);
}


void run_pool_forward_test() {
    printf("--> Perform pooling layer test with kernel size [2, 2]...\n");
    printf("Input:\n\n");

    uint8_t num_dim = 4;
    uint32_t *dim = (uint32_t *)malloc(num_dim * sizeof(uint32_t));
    dim[0] = 2; // # samples.
    dim[1] = 1; // # channels.
    dim[2] = 8; // Height.
    dim[3] = 6; // Width.
    Tensor *tensor = generate_test_random_4d_tensor(dim, 1, 0);
    print_tensor_4d(tensor);

    LayerGradients *grad = (LayerGradients *)malloc(sizeof(LayerGradients));

    printf("Max pool output:\n\n");
    run_pooling_forward(tensor, 2, MAX, grad);
    
    printf("Layer output:\n");
    print_tensor_4d(tensor);
    
    printf("Gradient:\n");
    print_tensor_4d(grad->dX_or_X);
    free_tensor(tensor);
    
    tensor = generate_test_random_4d_tensor(dim, 1, 0);
    printf("Mean pool output:\n\n");
    run_pooling_forward(tensor, 2, MEAN, grad);
    
    printf("Layer output:\n");
    print_tensor_4d(tensor);
    
    printf("Gradient:\n");
    print_tensor_4d(grad->dX_or_X);
    
    free_tensor(tensor);
    free_layer_gradients(grad);
}


void run_linear_layer_test() {
    printf("--> Perform linear layer test...\n");

    uint32_t feature1_height = 4;
    uint32_t feature1_width  = 5;
    uint32_t feature2_width  = 7;

    // Prepare feature 1.
    printf("Generating X:\n");
    Tensor *X = generate_test_ordered_tensor_2d(feature1_height, feature1_width, 1);
    printf("Generating A:\n");
    Tensor *A = generate_test_ordered_tensor_2d(feature2_width, feature1_width, 1);

    // Recall that the output will be stored in feature1.
    LayerGradients *gradients = (LayerGradients *)malloc(sizeof(LayerGradients));
    run_linear_forward(X, A, gradients);
    free_tensor(A);

    uint32_t out_height = feature1_height;
    uint32_t out_width  = feature2_width;
    float *values = (float *)malloc(out_height * out_width * sizeof(float));
    cudaMemcpy(values, X->values_d, out_height * out_width * sizeof(float), cudaMemcpyDeviceToHost);
    free_tensor(X);

    printf("Output: \n");
    for (uint32_t row = 0; row < out_height; ++row) {
        for (uint32_t col = 0; col < out_width; ++col) {
            printf("%8.0f", values[row * out_width + col]);
        }
        printf("\n");
    }
    printf("\n");
    free(values);

    printf("Gradients:\n");
    printf("dW\n");
    uint32_t dW_size = get_tensor_values_size(gradients->dW_or_W->num_dim, gradients->dW_or_W->dim);
    float *dW_values = (float *)malloc(dW_size * sizeof(float));
    cudaMemcpy(dW_values, gradients->dW_or_W->values_d, dW_size * sizeof(float), cudaMemcpyDeviceToHost);
    for (uint32_t row = 0; row < gradients->dW_or_W->dim[0]; ++row) {
        for (uint32_t col = 0; col < gradients->dW_or_W->dim[1]; ++col) {
            printf("%10.2f", dW_values[row * gradients->dW_or_W->dim[1] + col]);
        }
        printf("\n");
    }

    printf("\ndX\n");
    uint32_t dX_size = get_tensor_values_size(gradients->dX_or_X->num_dim, gradients->dX_or_X->dim);
    float *dX_values = (float *)malloc(dX_size * sizeof(float));
    cudaMemcpy(dX_values, gradients->dX_or_X->values_d, dX_size * sizeof(float), cudaMemcpyDeviceToHost);
    for (uint32_t row = 0; row < gradients->dX_or_X->dim[0]; ++row) {
        for (uint32_t col = 0; col < gradients->dX_or_X->dim[1]; ++col) {
            printf("%8.3f", dX_values[row * gradients->dX_or_X->dim[1] + col]);
        }
        printf("\n");
    }
    printf("\n");
}


void run_softmax_and_negative_log_likelihood_loss_test() {
    printf("--> Perform softmax and nll loss test...\n");
    printf("Input (X):\n");

    // Build X.
    uint32_t num_samples  = 5;
    uint32_t num_labels = 10;
    Tensor *X = generate_test_ordered_tensor_2d(num_samples, num_labels, 0.01);

    // Build y.
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
    free(y);

    // Run softmax.
    LayerGradients *gradients = (LayerGradients *)malloc(sizeof(LayerGradients));
    run_softmax_forward(X, y_d, gradients);

    float *values = (float *)malloc(num_samples * num_labels * sizeof(float));
    cudaMemcpy(values, X->values_d, num_samples * num_labels * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Softmax results: \n");
    for (uint32_t row = 0; row < num_samples; ++row) {
        for (uint32_t col = 0; col < num_labels; ++col) {
            printf("%9.4f", values[row * num_labels + col]);
        }
        printf("\n");
    }
    printf("\n");
    free(values);

    // Run NLL loss.
    Tensor *loss = compute_negative_log_likelihood_log_lost(X, y_d);
    free_tensor(X);
    cudaFree(y_d);
    
    float *loss_h = (float *)malloc(sizeof(float));
    cudaMemcpy(loss_h, loss->values_d, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Loss output: %.3f\n\n", *loss_h);
    free(loss_h);

    // Print softmax gradients.
    float *dX = (float *)malloc(num_samples * num_labels * sizeof(float));
    cudaMemcpy(dX, gradients->dX_or_X->values_d, num_samples * num_labels * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Softmax gradients: \n");
    for (uint32_t row = 0; row < num_samples; ++row) {
        for (uint32_t col = 0; col < num_labels; ++col) {
            printf("%8.3f", dX[row * num_labels + col]);
        }
        printf("\n");
    }
    printf("\n\n");
    free(dX);

    free_layer_gradients(gradients);
}


int main() {
    test_transpose_matrix();

    test_matmul();

    print_conv2d_weight_init_example(1, 3, 3);
    
    run_conv2d_forward_test();

    run_sigmoid_forward_test();

    run_pool_forward_test();

    run_linear_layer_test();

    run_softmax_and_negative_log_likelihood_loss_test();
}