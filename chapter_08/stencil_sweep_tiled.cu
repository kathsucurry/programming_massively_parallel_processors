#include <stdio.h>
#include <cuda_runtime.h>


#define ORDER 1
// The number of points in the three-dimensional stencil.
#define NUM_POINT 1 + ORDER*2*3
#define BLOCK_SIZE 4
#define IN_TILE_DIM BLOCK_SIZE
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2*(ORDER))


/**
 *  Perform stencil sweep kernel with shared memory tiling, corresponds to Fig. 8.8.
 */
__global__
void StencilKernel(
    float* input_array,
    float* output_array,
    float* coefficients,
    unsigned int size
) {
    int out_z_index = blockIdx.z * OUT_TILE_DIM + threadIdx.z - ORDER;
    int out_y_index = blockIdx.y * OUT_TILE_DIM + threadIdx.y - ORDER;
    int out_x_index = blockIdx.x * OUT_TILE_DIM + threadIdx.x - ORDER;
    int out_index = out_z_index * size * size + out_y_index * size + out_x_index;

    __shared__ float input_shared[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];
    if (out_z_index >= 0 && out_z_index < size
        && out_y_index >= 0 && out_y_index < size
        && out_x_index >= 0 && out_x_index < size)
        input_shared[threadIdx.z][threadIdx.y][threadIdx.x] = input_array[out_index];

    __syncthreads();

    // Here, we simplify the boundary condition which will not be updated from iteration
    // to the next. Hence, only calculate the inner output grid points.
    if (out_z_index >= 1 && out_z_index < size - 1 
        && out_y_index >= 1 && out_y_index < size - 1
        && out_x_index >= 1 && out_x_index < size - 1)
        if (threadIdx.z >= 1 && threadIdx.z < IN_TILE_DIM - 1
            && threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM - 1
            && threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM - 1)
            output_array[out_index] = coefficients[0] * input_shared[threadIdx.z][threadIdx.y][threadIdx.x]
                                    + coefficients[1] * input_shared[threadIdx.z][threadIdx.y][threadIdx.x - 1]
                                    + coefficients[2] * input_shared[threadIdx.z][threadIdx.y][threadIdx.x + 1]
                                    + coefficients[3] * input_shared[threadIdx.z][threadIdx.y - 1][threadIdx.x]
                                    + coefficients[4] * input_shared[threadIdx.z][threadIdx.y + 1][threadIdx.x]
                                    + coefficients[5] * input_shared[threadIdx.z - 1][threadIdx.y][threadIdx.x]
                                    + coefficients[6] * input_shared[threadIdx.z + 1][threadIdx.y][threadIdx.x];
                                    
}


void runStencil(
    float* input_array_h,
    float* output_array_h,
    float* coefficients_h,
    unsigned int size
) {
    // Get size in bytes.
    size_t size_input = size * size * size * sizeof(float);
    size_t size_coefficients = NUM_POINT * sizeof(float);

    // Load and copy input_array and filter to device memory.
    float * input_array_d, * output_array_d, * coefficients_d;
    
    cudaMalloc((void***)&input_array_d, size_input);
    cudaMemcpy(input_array_d, input_array_h, size_input, cudaMemcpyHostToDevice);

    cudaMalloc((void***)&coefficients_d, size_coefficients);
    cudaMemcpy(coefficients_d, coefficients_h, size_coefficients, cudaMemcpyHostToDevice);

    cudaMalloc((void***)&output_array_d, size_input);

    // Invoke kernel.
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(
        ceil(size / (OUT_TILE_DIM * 1.0)),
        ceil(size / (OUT_TILE_DIM * 1.0)),
        ceil(size / (OUT_TILE_DIM * 1.0)));
    StencilKernel<<<dimGrid, dimBlock>>>(
        input_array_d,
        output_array_d,
        coefficients_d,
        size
    );

    // Copy the output matrix from the device memory.
    cudaMemcpy(output_array_h, output_array_d, size_input, cudaMemcpyDeviceToHost);

    // Free device vectors.
    cudaFree(input_array_d);
    cudaFree(coefficients_d);
    cudaFree(output_array_d);
}


int main() {
    // Define input array as a 8 x 8 x 8 array with input (1 .. 512) % 10.
    int size = 8;
    float input_array[size * size * size];

    for (int i = 1; i < size * size * size + 1; ++i) 
        input_array[i-1] = i % 10;


    // Define coefficients.
    float coefficients[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};

    float output_array[size * size * size];

    runStencil(
        input_array,
        output_array,
        coefficients,
        size
    );

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            for (int k = 0; k < size; ++k) {
                printf("%.0f ", output_array[i * size * size + j * size + k]);
            }
            printf("\n");
        }
        printf("\n\n");
    }

    return 0;
}