/**
 * TODO:
 * - Load the entire MNIST data at once instead of per sample
 * - When changing the size of the image, avoid changing the image raw data
 * - Ensure all malloc and cudamalloc are successful (i.e., check for any errors)
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


// Assume fixed LR for now.
#define LEARNING_RATE 0.005


void eval_model() {

}


/**
 * For simplicity, the model architecture is fixed for now.
 * 
 * The model architecture is as follows.
 * - Convolution layer: 16 filters, each 5 x 5 --> output dim = 5 x 28 x 28
 * - Sigmoid layer
 * - Pooling layer: kernel size 2 x 2 --> output dim = 14 x 14
 * - Flatten layer --> output dim = 3136
 * - Linear layer (3136 x 10)
 * - Softmax layer
 * 
 */
NetworkOutputs *forward_pass(
    float *X_d, uint8_t *y_d,
    NetworkWeights *network_weights_d,
    uint32_t image_height,
    uint32_t image_width,
    uint32_t num_samples
) {
    uint8_t num_layers_with_grads = 5;
    LayerGradients *gradients = (LayerGradients *)malloc(num_layers_with_grads * sizeof(LayerGradients));
    
    uint32_t *image_dim = (uint32_t *)malloc(3 * sizeof(uint32_t));
    Tensor *output = initialize_tensor(X_d, 3, image_dim);
    
    // Layer 0: 2D convolution layer.
    run_conv2d_forward(output, network_weights_d->conv2d_weight, num_samples, image_height, image_width, &gradients[0]);
    
    // Layer 1: Sigmoid activation.
    run_sigmoid_forward(output, &gradients[1]);

    // Layer 2: Max pooling layer.
    uint32_t pool_kernel_length = 2;
    pooling_type pool_type = MAX;
    run_pooling_forward(output, pool_kernel_length, pool_type, &gradients[2]);
    
    // Layer 3: Convert into 1D vector; no grads.
    run_flatten_layer(output);

    // Layer 4: Linear layer.
    run_linear_forward(output, network_weights_d->linear_weight, &gradients[3]);

    // Layer 5: Softmax layer.
    run_softmax_forward(output, y_d, &gradients[4]);

    NetworkOutputs *network_outputs = (NetworkOutputs *)malloc(sizeof(NetworkOutputs));
    network_outputs->gradients = gradients;
    network_outputs->output = output;

    return network_outputs;
}


void backward_pass(NetworkOutputs *network_outputs, NetworkWeights *network_weights) {
    // Update linear weights.
    // Update conv2d weights.
}

NetworkWeights *train_model(ImageDataset *dataset, uint32_t batch_size) {
    // uint32_t num_samples = dataset->num_samples;
    uint32_t num_samples = dataset->num_samples;

    // Perform simple dataset split into training and validation.
    // Another approach is to assess the label distribution and split based on that
    // assuming the test data distribution follows the training data distribution.
    uint32_t num_train_samples = DATASET_SPLIT_TRAIN_PROPORTION * num_samples;
    uint32_t num_valid_samples = (1 - DATASET_SPLIT_TRAIN_PROPORTION) * num_samples;

    ImageDataset *train = split_dataset(dataset, 0, num_train_samples);
    ImageDataset *valid = split_dataset(dataset, num_train_samples, num_train_samples + num_valid_samples);
    if (train == NULL || valid == NULL) {
        printf("Error in dataset split.");
        return NULL;
    }

    // Prepare the model architecture: conv -> sigmoid -> pooling -> flatten -> linear -> softmax.
    // Initialize conv and linear layer weights using device memory.
    NetworkWeights *network_weights = (NetworkWeights *)malloc(sizeof(NetworkWeights));
    network_weights->conv2d_weight = initialize_conv_layer_weights(1, 16, 5, 0);
    network_weights->linear_weight = initialize_linear_layer_weights(3136, 10, 1);

    uint32_t num_epochs = 5;
    uint32_t num_epochs_valid_iter = 2;
    uint32_t num_batches = ceil(num_train_samples * 1.0 / BATCH_SIZE);
    uint32_t image_height = train->images[0].height;
    uint32_t image_width = train->images[0].width;
    uint32_t image_size = image_height * image_width;
    
    float train_X[BATCH_SIZE * image_size];
    uint8_t train_y[BATCH_SIZE * LABEL_SIZE];

    float *train_X_d;
    uint8_t *train_y_d;

    cudaMalloc((void**)&train_X_d, BATCH_SIZE * image_size * sizeof(float));
    cudaMalloc((void**)&train_y_d, BATCH_SIZE * LABEL_SIZE * sizeof(uint8_t));

    // For each epoch, run forward pass, evaluate on validation (i.e., forward pass + assess), backward pass.
    for (uint32_t epoch_index = 0; epoch_index < num_epochs; ++epoch_index) {
        // Shuffle the training indices.
        shuffle_indices(train, epoch_index);
        for (uint32_t batch_index = 0; batch_index < num_batches; ++batch_index) {
            uint32_t num_samples_in_batch = min(num_train_samples - batch_index * BATCH_SIZE, BATCH_SIZE);
            
            // Fill the batch.
            prepare_batch(train_X, train_y, train, num_samples_in_batch);

            // Copy host variables to device memory.
            cudaMemcpy(train_X_d, train_X, num_samples_in_batch * image_size * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(train_y_d, train_y, num_samples_in_batch * sizeof(uint8_t), cudaMemcpyHostToDevice);

            NetworkOutputs *network_outputs = forward_pass(train_X_d, train_y_d, network_weights, image_height, image_width, num_samples_in_batch);
            Tensor *loss = compute_negative_log_likelihood_log_lost(network_outputs->output, train_y_d);
            backward_pass(network_outputs, network_weights);
            
        }
        break;
        if (epoch_index > 0 && epoch_index % num_epochs_valid_iter == 0) {
            // float *valid_X[BATCH_SIZE * image_size];
            // uint8_t valid_y[BATCH_SIZE];
            // float *valid_logits = forward_pass(valid_X, valid_y, network_weights, image_size, num_valid_samples);
            // Calculate loss.
            // Evaluate model.
        }
    }

    cudaFree(train_X_d);
    cudaFree(train_y_d);

    free_image_dataset(train);
    free_image_dataset(valid);

    return network_weights;
}


int main() {
    MNISTDataset *train_dataset = load_mnist_dataset(
        "../../Dataset/mnist/train-images-idx3-ubyte",
        "../../Dataset/mnist/train-labels-idx1-ubyte"
    );
    MNISTDataset *test_dataset = load_mnist_dataset(
        "../../Dataset/mnist/t10k-images-idx3-ubyte",
        "../../Dataset/mnist/t10k-labels-idx1-ubyte"
    );

    printf("# Samples in training set: %d\n", train_dataset->num_samples);
    printf("# Samples in test set: %d\n", test_dataset->num_samples);

    // Normalize the pixel values to [0..1].
    ImageDataset *transformed_train_dataset = add_padding(
        normalize_pixels(
            prepare_dataset(train_dataset)
        ),
        2
    );

    NetworkWeights *model_weights = train_model(transformed_train_dataset, BATCH_SIZE);

    // Run evaluation on the test set.

    free(model_weights);
}