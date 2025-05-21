#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 4
#define FILTER_RADIUS 1

// Declare the filter array that utilizes constant memory.
__constant__ float CONST_FILTER[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];


/**
 *  (as long as the sizes are valid for performing matrix multiplication).
 */
__global__
void Convolution3DBasicKernel(
    float* input_array,
    float* output_array,
    int radius,
    int dim_x_size,
    int dim_y_size,
    int dim_z_size
) {
    int out_x_index = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y_index = blockIdx.y * blockDim.y + threadIdx.y;
    int out_z_index = blockIdx.z * blockDim.z + threadIdx.z;
    int out_index = out_z_index * dim_x_size * dim_y_size + out_y_index * dim_x_size + out_x_index;
    float out_value = 0.0f;

    for (int conv_z_index = -radius; conv_z_index < radius+1; conv_z_index++) {
        for (int conv_y_index = -radius; conv_y_index < radius+1; conv_y_index++) {
            for (int conv_x_index = -radius; conv_x_index < radius+1; conv_x_index++) {
                // Obtain the index of the input array.
                int in_z_index = out_z_index + conv_z_index;
                int in_y_index = out_y_index + conv_y_index;
                int in_x_index = out_x_index + conv_x_index;

                // Obtain the index of the filter.
                int filter_z_index = conv_z_index + radius;
                int filter_y_index = conv_y_index + radius;
                int filter_x_index = conv_x_index + radius;

                // Check the boundary.
                if (
                    in_z_index >= 0 && in_z_index < dim_z_size
                    && in_y_index >= 0 && in_y_index < dim_y_size
                    && in_x_index >= 0 && in_x_index < dim_x_size) {
                    int in_index = in_z_index * dim_x_size * dim_y_size + in_y_index * dim_x_size + in_x_index;
                    out_value += CONST_FILTER[filter_z_index][filter_y_index][filter_x_index]*input_array[in_index];
                }
            }
        }
    }
    output_array[out_index] = out_value;
}


void runConvolution3D(
    float* input_array_h,
    float* filter_h,
    float* output_array_h,
    int radius,
    int dim_x_size,
    int dim_y_size,
    int dim_z_size
) {
    // Get size in bytes.
    size_t size_input = dim_x_size * dim_y_size * dim_z_size * sizeof(float);
    size_t size_filter = (2*radius+1) * (2*radius+1) * (2*radius+1) * sizeof(float);

    // Load and copy input_array and filter to device memory.
    float * input_array_d, * output_array_d;
    
    cudaMalloc((void***)&input_array_d, size_input);
    cudaMemcpy(input_array_d, input_array_h, size_input, cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(CONST_FILTER, filter_h, size_filter);

    cudaMalloc((void***)&output_array_d, size_input);

    // Invoke kernel.
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(
        ceil(dim_x_size / (BLOCK_SIZE * 1.0)),
        ceil(dim_y_size / (BLOCK_SIZE * 1.0)),
        ceil(dim_z_size / (BLOCK_SIZE * 1.0)));
    Convolution3DBasicKernel<<<dimGrid, dimBlock>>>(
        input_array_d,
        output_array_d,
        radius,
        dim_x_size,
        dim_y_size,
        dim_z_size
    );

    // Copy the output matrix from the device memory.
    cudaMemcpy(output_array_h, output_array_d, size_input, cudaMemcpyDeviceToHost);

    // Free device vectors.
    cudaFree(input_array_d);
    cudaFree(output_array_d);
}


int main() {
    // Define input array as a 4 x 5 x 6 array with input 1 .. 120.
    int dim_x_size = 4, dim_y_size = 5, dim_z_size = 6;
    float input_array[dim_x_size * dim_y_size * dim_z_size];

    for (int i = 1; i < dim_x_size * dim_y_size * dim_z_size + 1; ++i) 
        input_array[i-1] = i;

    // Define filter as a 3 x 3 x 3 array with input 1 .. 27.
    int filter_dim_size = 2 * FILTER_RADIUS + 1;
    float filter[filter_dim_size * filter_dim_size * filter_dim_size];
    for (int i = 1; i < filter_dim_size * filter_dim_size * filter_dim_size + 1; ++i)
        filter[i-1] = i;

    float output_array[dim_x_size * dim_y_size * dim_z_size];

    runConvolution3D(
        input_array,
        filter,
        output_array,
        FILTER_RADIUS,
        dim_x_size,
        dim_y_size,
        dim_z_size);

    for (int i = 0; i < dim_z_size; ++i) {
        for (int j = 0; j < dim_y_size; ++j) {
            for (int k = 0; k < dim_x_size; ++k) {
                printf("%.0f ", output_array[i * dim_x_size * dim_y_size + j * dim_x_size + k]);
            }
            printf("\n");
        }
        printf("\n\n");
    }

    return 0;
}