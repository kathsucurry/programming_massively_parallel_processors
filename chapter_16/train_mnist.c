/**
 * TODO:
 * - Load the entire MNIST data at once instead of per sample
 * - When changing the size of the image, avoid changing the image raw data
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "src/data_loader.h"
#include "src/preprocessing.h"
#include "src/common.h"
#include "src/cnn_layers.cuh"





/**
 * For simplicity, the model architecture is fixed.
 * 
 * The model architecture is as follows.
 * - Convolution layer: 16 filters, each 5 x 5 --> output dim = 28 x 28
 * - Sigmoid layer
 * - Pooling layer: kernel size 2 x 2 --> output dim = 14 x 14
 * - Flatten layer --> output dim = 3136
 * - Linear layer (3136 x 10)
 * - Softmax layer
 * 
 */
void forward_pass() {

}

void backward_pass() {}

void train_model(ImageDataset *dataset, uint32_t batch_size) {
    // uint32_t num_samples = dataset->num_samples;
    uint32_t num_samples = 20;
    dataset->num_samples = num_samples;

    // Perform simple dataset split into training and validation.
    // Another approach is to assess the label distribution and split based on that
    // assuming the test data distribution follows the training data distribution.
    uint32_t num_train_samples = DATASET_SPLIT_TRAIN_PROPORTION * num_samples;
    uint32_t num_valid_samples = (1 - DATASET_SPLIT_TRAIN_PROPORTION) * num_samples;

    ImageDataset *train = split_dataset(dataset, 0, num_train_samples);
    ImageDataset *valid = split_dataset(dataset, num_train_samples, num_train_samples + num_valid_samples);
    if (train == NULL || valid == NULL) {
        printf("Error in dataset split.");
        return;
    }

    // Prepare the model architecture: conv -> sigmoid -> pooling -> flatten -> linear -> softmax.
    // Initialize conv and linear layer weights.
    NetworkWeights *network_weights = malloc(sizeof(NetworkWeights));
    network_weights->conv2_weights = initialize_conv_layer_weights(1, 16, 5);
    network_weights->linear_weights = initialize_linear_layer_weights(3136, 10);

    // Define number of epochs.
    uint32_t num_epochs = 5;
    // For each epoch, run forward pass, evaluate on validation (i.e., forward pass + assess), backward pass.
    for (uint32_t epoch_index = 0; epoch_index < num_epochs; ++epoch_index) {
        // Get training loss.
        forward_pass();
        // Get validation loss and evaluation metrics.
        backward_pass();
    }

}

void eval_model() {
    forward_pass();
    // Perform evaluation.

}


int compareIntPointer(const void *a, const void *b) {
    return ( *(uint32_t*)a - *(uint32_t*)b );
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

    train_model(transformed_train_dataset, BATCH_SIZE);

    // Run evaluation on the test set.
}