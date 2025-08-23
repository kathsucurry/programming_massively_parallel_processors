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
#define POOL_TYPE MAX
#define NUM_EPOCHS 5
// The loss and accuracy of the validation set will be computed every NUM_VALID_ITER epochs.
#define NUM_EPOCHS_VALID_ITER 2 


ImageDataset *preprocess_images(MNISTDataset *mnist_dataset) {
    ImageDataset *transformed = add_padding(
        normalize_pixels(
            prepare_dataset(mnist_dataset)
        ),
        2
    );
    return transformed;
}


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
    run_pooling_forward(output, POOL_KERNEL_LENGTH, POOL_TYPE, &gradients[2], compute_grad);
    
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


EpochOutput run_one_epoch(
    ImageDataset *dataset,
    float X[], uint8_t y[],
    float *X_d, uint8_t *y_d,
    NetworkWeights *network_weights,
    bool update_weight,
    bool compute_accuracy
) {
    uint32_t num_samples = dataset->num_samples;
    uint32_t num_batches = ceil(num_samples * 1.0 / BATCH_SIZE);

    uint32_t image_height = dataset->images[0].height;
    uint32_t image_width = dataset->images[0].width;
    uint32_t image_size = image_height * image_width;

    float loss_sum = 0.0f;
    uint32_t correct_pred_sum = 0;
    EpochOutput epoch_output;

    for (uint32_t batch_index = 0; batch_index < num_batches; ++batch_index) {
        uint32_t num_samples_in_batch = min(num_samples - batch_index * BATCH_SIZE, BATCH_SIZE);

        // Fill the batch.
        prepare_batch(X, y, dataset, batch_index * BATCH_SIZE, num_samples_in_batch);

        // Copy host variables to device memory.
        cudaMemcpy(X_d, X, num_samples_in_batch * image_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(y_d, y, num_samples_in_batch * LABEL_SIZE * sizeof(uint8_t), cudaMemcpyHostToDevice);

        NetworkOutputs *network_outputs = forward_pass(
            X_d, y_d,
            network_weights,
            image_height, image_width,
            num_samples_in_batch,
            update_weight
        );
            
        float *loss = compute_negative_log_likelihood_log_lost(network_outputs->output, y_d);
        loss_sum += *loss;

        if (update_weight)
            backward_pass(network_outputs->gradients, network_weights, num_samples_in_batch, LEARNING_RATE);

        if (compute_accuracy)
            correct_pred_sum += 0;

        free(loss);
        free_network_outputs(network_outputs, update_weight);
    }

    epoch_output.loss = loss_sum;
    epoch_output.accuracy_percent = correct_pred_sum / num_samples * 100.0;

    return epoch_output;
}


NetworkWeights *train_model(ImageDataset *dataset) {
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

    uint32_t image_height = train->images[0].height;
    uint32_t image_width = train->images[0].width;
    uint32_t image_size = image_height * image_width;
    
    float X[BATCH_SIZE * image_size];
    uint8_t y[BATCH_SIZE * LABEL_SIZE];

    float *X_d;
    uint8_t *y_d;

    cudaMallocCheck((void**)&X_d, BATCH_SIZE * image_size * sizeof(float));
    cudaMallocCheck((void**)&y_d, BATCH_SIZE * LABEL_SIZE * sizeof(uint8_t));

    for (uint32_t epoch_index = 0; epoch_index < NUM_EPOCHS; ++epoch_index) {
        // Run training.
        shuffle_indices(train, epoch_index);
        EpochOutput train_epoch_output = run_one_epoch(
            train,
            X, y,
            X_d, y_d,
            network_weights,
            true,
            false
        );

        printf("Epoch loss is %.7f\n", train_epoch_output.loss);
        if (epoch_index > 0 && epoch_index % NUM_EPOCHS_VALID_ITER == 0) {
            // Run forward pass on validation set.
            EpochOutput valid_epoch_output = run_one_epoch(
                valid,
                X, y,
                X_d, y_d,
                network_weights,
                false,
                true
            );
        }
    }
    
    cudaFree(X_d);
    cudaFree(y_d);

    free_dataset(train);
    free_dataset(valid);

    return network_weights;
}


int main() {
    MNISTDataset *dataset = load_mnist_dataset(
        "../../Dataset/mnist/train-images-idx3-ubyte",
        "../../Dataset/mnist/train-labels-idx1-ubyte"
    );
    printf("[INFO] # Samples in training set: %d\n", dataset->num_samples);

    ImageDataset *transformed_train_dataset = preprocess_images(dataset);
    free_MNIST_dataset(dataset);

    NetworkWeights *model_weights = train_model(transformed_train_dataset);
    free_dataset(transformed_train_dataset);

    // Run evaluation on the test set.
    // MNISTDataset *test_dataset = load_mnist_dataset(
    //     "Study/Dataset/mnist/t10k-images-idx3-ubyte",
    //     "Study/Dataset/mnist/t10k-labels-idx1-ubyte"
    // );

    free_network_weights(model_weights);
}