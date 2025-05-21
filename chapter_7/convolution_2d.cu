#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 4
#define FILTER_RADIUS 1

// Declare the filter array that utilizes constant memory.
__constant__ float CONST_FILTER[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];


/**
 *  (as long as the sizes are valid for performing matrix multiplication).
 */
__global__
void Convolution2DBasicKernel(
    float* input_array,
    float* output_array,
    int radius,
    int width,
    int height
) {
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    float out_value = 0.0f;

    for (int conv_row = -radius; conv_row < radius+1; conv_row++) {
        for (int conv_col = -radius; conv_col < radius+1; conv_col++) {
            // Obtain the index of the input array.
            int input_row = out_row + conv_row;
            int input_col = out_col + conv_col;

            // Obtain the index of the filter.
            int filter_row = conv_row + radius;
            int filter_col = conv_col + radius;

            // Check the boundary.
            if (input_row >= 0 && input_row < height && input_col >= 0 && input_col < width) {
                int input_index = input_row * width + input_col;
                out_value += CONST_FILTER[filter_row][filter_col]*input_array[input_index];
            }
        }
    }

    output_array[out_row*width+out_col] = out_value;
}


void runConvolution2D(
    float* input_array_h,
    float* filter_h,
    float* output_array_h,
    int radius,
    int width,
    int height
) {
    // Get size in bytes.
    size_t size_input = width * height * sizeof(float);
    size_t size_filter = (2*radius+1) * (2*radius+1) * sizeof(float);

    // Load and copy input_array and filter to device memory.
    float * input_array_d, * output_array_d;
    
    cudaMalloc((void***)&input_array_d, size_input);
    cudaMemcpy(input_array_d, input_array_h, size_input, cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(CONST_FILTER, filter_h, size_filter);

    cudaMalloc((void***)&output_array_d, size_input);

    // Invoke kernel.
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(ceil(width / (BLOCK_SIZE * 1.0)), ceil(height / (BLOCK_SIZE * 1.0)));
    Convolution2DBasicKernel<<<dimGrid, dimBlock>>>(
        input_array_d,
        output_array_d,
        radius,
        width,
        height
    );

    // Copy the output matrix from the device memory.
    cudaMemcpy(output_array_h, output_array_d, size_input, cudaMemcpyDeviceToHost);

    // Free device vectors.
    cudaFree(input_array_d);
    cudaFree(output_array_d);
}


int main() {
    // Define input array as a 4 x 3 array with values 1 .. 12.
    float input_array[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    int height = 4, width = 3;

    // Define the filter as a 3 x 3 array with values 1 .. 9.
    float filter[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    int radius = 1;

    float output_array[12];

    runConvolution2D(
        input_array,
        filter,
        output_array,
        radius,
        width,
        height);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            printf("%.0f ", output_array[i * width + j]);
        }
        printf("\n");
    }

    return 0;
}