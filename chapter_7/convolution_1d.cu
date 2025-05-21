#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 4
#define FILTER_RADIUS 1

// Declare the filter array that utilizes constant memory.
__constant__ float CONST_FILTER[2*FILTER_RADIUS+1];


/**
 *  (as long as the sizes are valid for performing matrix multiplication).
 */
__global__
void Convolution1DBasicKernel(
    float* input_array,
    float* output_array,
    int radius,
    int length
) {
    int out_index = blockIdx.x * blockDim.x + threadIdx.x;
    float out_value = 0.0f;

    for (int conv_index = -radius; conv_index < radius+1; conv_index++) {
        // Obtain the index of the input array.
        int input_index = out_index + conv_index;

        // Obtain the index of the filter.
        int filter_index = conv_index + radius;

        // Check the boundary.
        if (input_index >= 0 && input_index < length) {
            out_value += CONST_FILTER[filter_index]*input_array[input_index];
        }
    }

    output_array[out_index] = out_value;
}


void runConvolution1D(
    float* input_array_h,
    float* filter_h,
    float* output_array_h,
    int radius,
    int length
) {
    // Get size in bytes.
    size_t size_input = length * sizeof(float);
    size_t size_filter = (2*radius+1) * sizeof(float);

    // Load and copy input_array and filter to device memory.
    float * input_array_d, * output_array_d;
    
    cudaMalloc((void***)&input_array_d, size_input);
    cudaMemcpy(input_array_d, input_array_h, size_input, cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(CONST_FILTER, filter_h, size_filter);

    cudaMalloc((void***)&output_array_d, size_input);

    // Invoke kernel.
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(ceil(length / (BLOCK_SIZE * 1.0)));
    Convolution1DBasicKernel<<<dimGrid, dimBlock>>>(
        input_array_d,
        output_array_d,
        radius,
        length
    );

    // Copy the output matrix from the device memory.
    cudaMemcpy(output_array_h, output_array_d, size_input, cudaMemcpyDeviceToHost);

    // Free device vectors.
    cudaFree(input_array_d);
    cudaFree(output_array_d);
}


int main() {
    // Define a 1Dinput array with values 1 .. 12.
    float input_array[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    int length = 12;

    // Define a 1 x 3 filter array with values 1 .. 3.
    float filter[] = {1, 2, 3};

    float output_array[12];

    runConvolution1D(
        input_array,
        filter,
        output_array,
        FILTER_RADIUS,
        length);

    for (int i = 0; i < length; ++i) {
        printf("%.0f ", output_array[i]);
    }

    return 0;
}