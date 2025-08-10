#include <stdio.h>
#include "src/mnist_read.h"
// #include "src/preprocessing.h"

int main() {
    MNISTDataset *train_dataset = load_mnist_dataset(
        "../../Dataset/mnist/train-images-idx3-ubyte",
        "../../Dataset/mnist/train-labels-idx1-ubyte"
    );
    MNISTDataset *test_dataset = load_mnist_dataset(
        "../../Dataset/mnist/t10k-images-idx3-ubyte",
        "../../Dataset/mnist/t10k-labels-idx1-ubyte"
    );

    printf("#Samples in training set: %d\n", train_dataset->num_samples);
    printf("#Samples in test set: %d\n", test_dataset->num_samples);

    // Normalize the pixel values to [0..1].
    // normalize_pixels(train_dataset);


    // zero padding to make it 32 x 32?

    // Define model architecture.
    // conv -> pooling -> conv -> pooling -> full connection -> output

    // For each epoch:
    // - forward pass
    // - calculate loss
    // - back prop
    // - run inference for validation set --> get metric (loss + f1)
}