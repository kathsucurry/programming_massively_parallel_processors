/**
 * For simplicity, the model architecture is fixed for now.
 * 
 * The model architecture is as follows.
 * - Convolution layer: 16 filters, each 5 x 5 --> output dim = 16 x 28 x 28
 * - Sigmoid layer
 * - Pooling layer: kernel size 2 x 2 --> output dim = 14 x 14
 * - Flatten layer --> output dim = 3136
 * - Linear layer (3136 x 10)
 * - Softmax layer
 * 
 * TODO:
 * - Load the entire MNIST data at once instead of per sample
 * - When adding paddings to images, avoid changing the image raw data directly (i.e., change the view instead)
 * - Ensure all malloc and cudamalloc are successful (i.e., check for any errors)
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#include "src/data_loader.cuh"
#include "src/preprocessing.cuh"
#include "src/common.cuh"
#include "src/cnn_layers.cuh"
#include "tests/test_utils.cuh"


#define LEARNING_RATE 0.01
#define POOL_KERNEL_LENGTH 2


NetworkOutputs *forward_pass(
    float *X_d, uint8_t *y_d,
    NetworkWeights *network_weights_d,
    uint32_t image_height,
    uint32_t image_width,
    uint32_t num_samples,
    bool compute_grad
) {
    uint8_t num_layers_with_grads = 6;
    LayerGradients *gradients = (LayerGradients *)mallocCheck(num_layers_with_grads * sizeof(LayerGradients));
    
    uint32_t *image_dim = (uint32_t *)mallocCheck(4 * sizeof(uint32_t));
    image_dim[0] = num_samples;
    image_dim[1] = 1; // Number of channels.
    image_dim[2] = image_height;
    image_dim[3] = image_width;
    Tensor *output = initialize_tensor(X_d, 4, image_dim);
    
    // Layer 0: 2D convolution layer.
    run_conv2d_forward(output, network_weights_d->conv2d_weight, &gradients[0], compute_grad);

    // Layer 1: Sigmoid activation.
    run_sigmoid_forward(output, &gradients[1], compute_grad);

    // Layer 2: Max pooling layer.
    uint32_t pool_kernel_length = POOL_KERNEL_LENGTH;
    pooling_type pool_type = MAX;
    run_pooling_forward(output, pool_kernel_length, pool_type, &gradients[2], compute_grad);
    
    // Layer 3: Convert into 1D vector; no grads created.
    run_flatten_forward(output);

    // Layer 4: Linear layer.
    run_linear_forward(output, network_weights_d->linear_weight, &gradients[4], compute_grad);

    // Layer 5: Softmax layer.
    run_softmax_forward(output, y_d, &gradients[5], compute_grad);

    NetworkOutputs *network_outputs = (NetworkOutputs *)mallocCheck(sizeof(NetworkOutputs));
    network_outputs->gradients = gradients;
    network_outputs->output = output;
    network_outputs->num_layers = 6;

    return network_outputs;
}


void backward_pass(LayerGradients *gradients, NetworkWeights *network_weights, uint32_t num_samples, float learning_rate) {
    // Go through layers from the second last to the first to update gradients + weights.
    
    // Layer 4: linear layer - update both gradients and weights.
    run_linear_backward(network_weights->linear_weight, &gradients[4], &gradients[5], learning_rate);

    // Layer 3: flatten layer (i.e., change the dimension of the next layer's gradients).
    run_flatten_backward(num_samples, POOL_KERNEL_LENGTH, &gradients[3], &gradients[4]);

    // Layer 2: pooling layer.
    run_pooling_backward(POOL_KERNEL_LENGTH, &gradients[2], &gradients[3]);

    // Layer 1: sigmoid layer.
    run_sigmoid_backward(&gradients[1], &gradients[2]);
    
    // Layer 0: conv2d layer - update both gradients and weights.
    run_conv2d_backward(network_weights->conv2d_weight, &gradients[0], &gradients[1], learning_rate);
}

NetworkWeights *train_model(ImageDataset *dataset, const uint32_t batch_size) {
    uint32_t num_samples = dataset->num_samples;

    // Perform simple dataset split into training and validation.
    // Another approach is to assess the label distribution and split based on that
    // assuming the test data distribution follows the training data distribution.
    uint32_t num_train_samples = DATASET_SPLIT_TRAIN_PROPORTION * num_samples;
    uint32_t num_valid_samples = (1 - DATASET_SPLIT_TRAIN_PROPORTION) * num_samples;
    printf("The train dataset is splitted into training (n=%u) and validation (n=%u).\n", num_train_samples, num_valid_samples);

    ImageDataset *train = split_dataset(dataset, 0, num_train_samples, false);
    ImageDataset *valid = split_dataset(dataset, num_train_samples, num_train_samples + num_valid_samples, true);

    if (train == NULL || valid == NULL) {
        printf("Error in dataset split.");
        return NULL;
    }

    // Prepare the model architecture that includes one conv2d and one linear layers.
    // Initialize conv and linear layer weights using device memory.
    NetworkWeights *network_weights = (NetworkWeights *)mallocCheck(sizeof(NetworkWeights));
    network_weights->conv2d_weight = initialize_conv_layer_weights(1, 16, 5, 0);
    network_weights->linear_weight = initialize_linear_layer_weights(3136, 10, 1);

    uint32_t num_epochs = 5;
    uint32_t num_epochs_valid_iter = 2;
    uint32_t num_batches = ceil(num_train_samples * 1.0 / batch_size);
    printf("# Batches: %u\n", num_batches);

    uint32_t image_height = train->images[0].height;
    uint32_t image_width = train->images[0].width;
    uint32_t image_size = image_height * image_width;
    
    float train_X[batch_size * image_size];
    uint8_t train_y[batch_size * LABEL_SIZE];

    float *train_X_d;
    uint8_t *train_y_d;

    cudaMallocCheck((void**)&train_X_d, batch_size * image_size * sizeof(float));
    cudaMallocCheck((void**)&train_y_d, batch_size * LABEL_SIZE * sizeof(uint8_t));

    for (uint32_t epoch_index = 0; epoch_index < num_epochs; ++epoch_index) {
        float epoch_loss = 0;    
    
        shuffle_indices(train, epoch_index);

        for (uint32_t batch_index = 0; batch_index < num_batches; ++batch_index) {
            uint32_t num_samples_in_batch = min(num_train_samples - batch_index * batch_size, batch_size);

            // Fill the batch.
            prepare_batch(train_X, train_y, train, batch_index * batch_size, num_samples_in_batch);

            // Copy host variables to device memory.
            cudaMemcpy(train_X_d, train_X, num_samples_in_batch * image_size * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(train_y_d, train_y, num_samples_in_batch * LABEL_SIZE * sizeof(uint8_t), cudaMemcpyHostToDevice);

            NetworkOutputs *network_outputs = forward_pass(
                train_X_d, train_y_d,
                network_weights,
                image_height, image_width,
                num_samples_in_batch,
                true
            );
            
            float *loss = compute_negative_log_likelihood_log_lost(network_outputs->output, train_y_d);
            epoch_loss += *loss;

            backward_pass(network_outputs->gradients, network_weights, num_samples_in_batch, LEARNING_RATE);

            free(loss);
            free_network_outputs(network_outputs, true);
        }

        printf("Epoch loss is %.7f\n", epoch_loss);
        if (epoch_index > 0 && epoch_index % num_epochs_valid_iter == 0) {
            // float *valid_X[batch_size * image_size];
            // uint8_t valid_y[batch_size];
            // float *valid_logits = forward_pass(valid_X, valid_y, network_weights, image_size, num_valid_samples);
            // Calculate loss.
            // Evaluate model.
        }
    }
    
    cudaFree(train_X_d);
    cudaFree(train_y_d);

    free_dataset(train);
    free_dataset(valid);

    return network_weights;
}


int main() {
    MNISTDataset *dataset = load_mnist_dataset(
        "../../Dataset/mnist/train-images-idx3-ubyte",
        "../../Dataset/mnist/train-labels-idx1-ubyte"
    );

    printf("# Samples in training set: %d\n", dataset->num_samples);

    // Normalize the pixel values to [0..1].
    ImageDataset *transformed_train_dataset = add_padding(
        normalize_pixels(
            prepare_dataset(dataset)
        ),
        2
    );

    free_MNIST_dataset(dataset);

    // print_sample(transformed_train_dataset, 2);

    NetworkWeights *model_weights = train_model(transformed_train_dataset, BATCH_SIZE);

    // Run evaluation on the test set.
    // MNISTDataset *test_dataset = load_mnist_dataset(
    //     "Study/Dataset/mnist/t10k-images-idx3-ubyte",
    //     "Study/Dataset/mnist/t10k-labels-idx1-ubyte"
    // );

    free_network_weights(model_weights);
    free_dataset(transformed_train_dataset);
}