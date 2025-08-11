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
#include "src/cnn_layers.h"


/**
 * For simplicity, the model architecture is fixed.
 */
void forward_pass() {

}

void backward_pass() {}

void train_model(ImageDataset *dataset) {
    uint32_t num_samples = dataset->num_samples;

    // Perform simple dataset split into training and validation.
    // Another approach is to assess the label distribution and split based on that
    // assuming the test data distribution follows the training data distribution.
    uint32_t num_train_samples = DATASET_SPLIT_TRAIN_PROPORTION * num_samples;
    uint32_t num_valid_samples = (1 - DATASET_SPLIT_TRAIN_PROPORTION) * num_samples;

    uint8_t seed = 0;
    uint32_t *shuffled_indices = shuffle_indices(num_samples, seed);

    ImageDataset *train = split_dataset(dataset, shuffled_indices, num_train_samples);
    ImageDataset *valid = split_dataset(dataset, &shuffled_indices[num_train_samples], num_valid_samples);
    if (train == NULL || valid == NULL) {
        printf("Error in dataset split.");
        return;
    }

    // Prepare the model architecture: conv -> sigmoid -> pooling -> flatten -> linear -> softmax.
    // Initialize conv and linear layer weights.
    float *test = _uniform_xavier_initialization_1d(25, 75, 10);
    for (int i = 0; i < 10; ++i)
        printf("%.2f ", test[i]);
    printf("\n");

    // Initialize network weights.


    // Define number of epochs.
    // For each epoch, run forward pass, evaluate on validation (i.e., forward pass + assess), backward pass.
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

    train_model(transformed_train_dataset);

    // Run evaluation on the test set.
}