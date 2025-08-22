/**
 * Performs small tests with the following steps:
 * 1) Load and store only the first five images.
 * 2) Run 1 batch (size = 3) iteration of forward and backward pass.
 * 3) Check results along the way.
 * 
 * The model architecture is identical to the one used in train_mnist.cu:
 * - Convolution layer: 4 filters, each 5 x 5 --> output dim = 4 x 28 x 28
 * - Sigmoid layer
 * - Pooling layer: kernel size 2 x 2 --> output dim = 14 x 14
 * - Flatten layer --> output dim = 784 (4 filters x 14 x 14)
 * - Linear layer (784 x 10)
 * - Softmax layer
 * 
 */

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


#define LEARNING_RATE 0.005
#define POOL_KERNEL_LENGTH 2
#define NUM_SAMPLES 5


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
    compare_tensor("layer 0 conv2d", "tests/outputs/output_layer0_conv2d.txt", output);

    // Layer 1: Sigmoid activation.
    run_sigmoid_forward(output, &gradients[1], compute_grad);
    compare_tensor("layer 1 sigmoid", "tests/outputs/output_layer1_sigmoid.txt", output);

    // Layer 2: Max pooling layer.
    uint32_t pool_kernel_length = POOL_KERNEL_LENGTH;
    pooling_type pool_type = MAX;
    run_pooling_forward(output, pool_kernel_length, pool_type, &gradients[2], compute_grad);
    compare_tensor("layer 2 maxpool", "tests/outputs/output_layer2_maxpool.txt", output);
    
    // Layer 3: Convert into 1D vector; no grads created.
    run_flatten_forward(output);
    compare_tensor("layer 3 flatten", "tests/outputs/output_layer3_flatten.txt", output);

    // Layer 4: Linear layer.
    run_linear_forward(output, network_weights_d->linear_weight, &gradients[4], compute_grad);
    compare_tensor("layer 4 linear", "tests/outputs/output_layer4_linear.txt", output);

    // Layer 5: Softmax layer.
    run_softmax_forward(output, y_d, &gradients[5], compute_grad);
    compare_tensor("layer 5 softmax", "tests/outputs/output_layer5_softmax.txt", output);

    NetworkOutputs *network_outputs = (NetworkOutputs *)mallocCheck(sizeof(NetworkOutputs));
    network_outputs->gradients = gradients;
    network_outputs->output = output;
    network_outputs->num_layers = 6;

    return network_outputs;
}


void backward_pass(LayerGradients *gradients, NetworkWeights *network_weights, uint32_t num_samples, float learning_rate) {
    // Go through layers from the second last to the first to update gradients + weights.
    compare_tensor("layer 4 linear dY", "tests/outputs/dy_layer4_linear.txt", gradients[5].dX_or_X);
    
    // Layer 4: linear layer - update both gradients and weights.
    run_linear_backward(network_weights->linear_weight, &gradients[4], &gradients[5], learning_rate);
    compare_tensor("layer 4 linear dX", "tests/outputs/dy_layer3_flatten.txt", gradients[4].dX_or_X);
    compare_tensor("layer 4 linear dW", "tests/outputs/weight_grad_layer4_linear.txt", gradients[4].dW_or_W);
    compare_tensor("layer 4 linear updated W", "tests/outputs/updated_weight_layer4_linear.txt", network_weights->linear_weight);

    // Layer 3: flatten layer (i.e., change the dimension of the next layer's gradients).
    run_flatten_backward(num_samples, POOL_KERNEL_LENGTH, &gradients[3], &gradients[4]);
    compare_tensor("layer 3 flatten dX", "tests/outputs/dy_layer2_maxpool.txt", gradients[3].dX_or_X);

    // Layer 2: pooling layer.
    run_pooling_backward(POOL_KERNEL_LENGTH, &gradients[2], &gradients[3]);
    compare_tensor("layer 2 maxpool dX", "tests/outputs/dy_layer1_sigmoid.txt", gradients[2].dX_or_X);

    // Layer 1: sigmoid layer.
    run_sigmoid_backward(&gradients[1], &gradients[2]);
    compare_tensor("layer 1 sigmoid dX", "tests/outputs/dy_layer0_conv2d.txt", gradients[1].dX_or_X);
    
    // Layer 0: conv2d layer - update both gradients and weights.
    run_conv2d_backward(network_weights->conv2d_weight, &gradients[0], &gradients[1], learning_rate);
    compare_tensor("layer 0 conv2d dW", "tests/outputs/weight_grad_layer0_conv2d.txt", gradients[0].dW_or_W);
    compare_tensor("layer 0 conv2d updated W", "tests/outputs/updated_weight_layer0_conv2d.txt", network_weights->conv2d_weight);
}

NetworkWeights *train_model(ImageDataset *dataset, uint32_t batch_size) {
    // Prepare the model architecture: conv -> sigmoid -> pooling -> flatten -> linear -> softmax.
    // Initialize conv and linear layer weights using device memory.
    NetworkWeights *network_weights = (NetworkWeights *)mallocCheck(sizeof(NetworkWeights));
    network_weights->conv2d_weight = initialize_conv_layer_weights(1, 4, 5, 0);
    test_initialize_conv_layer_weights(network_weights->conv2d_weight);
    
    network_weights->linear_weight = initialize_linear_layer_weights(784, 10, 1);
    test_initialize_linear_layer_weights(network_weights->linear_weight);

    uint32_t image_height = dataset->images[0].height;
    uint32_t image_width = dataset->images[0].width;
    uint32_t image_size = image_height * image_width;
    
    float train_X[batch_size * image_size];
    uint8_t train_y[batch_size * LABEL_SIZE];

    float *train_X_d;
    uint8_t *train_y_d;

    cudaMallocCheck((void**)&train_X_d, batch_size * image_size * sizeof(float));
    cudaMallocCheck((void**)&train_y_d, batch_size * LABEL_SIZE * sizeof(uint8_t));

    shuffle_indices(dataset, 0);
    test_shuffle_indices(dataset);

    // Fill the batch.
    prepare_batch(train_X, train_y, dataset, 0, batch_size);
    test_prepare_batch(train_X, train_y, dataset, false);

    // Copy host variables to device memory.
    cudaMemcpy(train_X_d, train_X, batch_size * image_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(train_y_d, train_y, batch_size * LABEL_SIZE * sizeof(uint8_t), cudaMemcpyHostToDevice);

    NetworkOutputs *network_outputs = forward_pass(train_X_d, train_y_d, network_weights, image_height, image_width, batch_size, true);
    
    float *loss = compute_negative_log_likelihood_log_lost(network_outputs->output, train_y_d);
    compare_float("cross entropy loss", "tests/outputs/output_loss.txt", loss);

    backward_pass(network_outputs->gradients, network_weights, batch_size, LEARNING_RATE);

    free(loss);
    free_network_outputs(network_outputs, true);
    cudaFree(train_X_d);
    cudaFree(train_y_d);

    return network_weights;
}


int main() {
    MNISTDataset *dataset = load_mnist_dataset(
        "../../Dataset/mnist/train-images-idx3-ubyte",
        "../../Dataset/mnist/train-labels-idx1-ubyte"
    );

    // Keep only the first three samples.
    keep_first_n_samples(dataset, NUM_SAMPLES);

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

    NetworkWeights *model_weights = train_model(transformed_train_dataset, 3);

    // Run evaluation on the test set.

    free_dataset(transformed_train_dataset);
    free_network_weights(model_weights);
}