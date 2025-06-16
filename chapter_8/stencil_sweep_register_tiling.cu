#include <stdio.h>
#include <cuda_runtime.h>


#define ORDER 1
// The number of points in the three-dimensional stencil.
#define NUM_POINT 1 + ORDER*2*3
#define BLOCK_SIZE 4
#define IN_TILE_DIM BLOCK_SIZE
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2*(ORDER))


/**
 *  Perform stencil sweep kernel with thread coarsening in the z direction + register tiling, corresponds to Fig. 8.12.
 */
__global__
void StencilKernel(
    float* input_array,
    float* output_array,
    float* coefficients,
    unsigned int size
) {
    int z_index_start = blockIdx.z * OUT_TILE_DIM;
    int y_index = blockIdx.y * OUT_TILE_DIM + threadIdx.y - ORDER;
    int x_index = blockIdx.x * OUT_TILE_DIM + threadIdx.x - ORDER;

    // Only 3 layers are needed since the order is 1.
    float input_prev_float;
    __shared__ float input_curr[IN_TILE_DIM][IN_TILE_DIM];
    float input_next_float;

    // Store the first layer in the shared memory.
    if (z_index_start-1 >= 0 && z_index_start-1 < size
        && y_index >= 0 && y_index < size
        && x_index >= 0 && x_index < size)
        input_prev_float = input_array[(z_index_start-1)*size*size + y_index*size + x_index];

    // Store the second layer in the shared memory.
    if (z_index_start >= 0 && z_index_start < size
        && y_index >= 0 && y_index < size
        && x_index >= 0 && x_index < size)
        input_curr[threadIdx.y][threadIdx.x] = input_array[z_index_start*size*size + y_index*size + x_index];


    // Iterate the third layer through the rest of the tile dimension.
    for (int z_index = z_index_start; z_index < z_index_start + OUT_TILE_DIM; ++z_index) {
        // Store the third layer in the shared memory.
        if (z_index+1 >= 0 && z_index+1 < size
            && y_index >= 0 && y_index < size
            && x_index >= 0 && x_index < size)
            input_next_float = input_array[(z_index+1)*size*size + y_index*size + x_index];
            
        __syncthreads();

        unsigned int out_index = z_index*size*size + y_index*size + x_index;

        // Compute output.
        if (z_index >= 1 && z_index < size-1 
            && y_index >= 1 && y_index < size-1
            && x_index >= 1 && x_index < size-1)
            if (threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM - 1
                && threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM - 1)
                output_array[out_index] = coefficients[0] * input_curr[threadIdx.y][threadIdx.x]
                                        + coefficients[1] * input_curr[threadIdx.y][threadIdx.x - 1]
                                        + coefficients[2] * input_curr[threadIdx.y][threadIdx.x + 1]
                                        + coefficients[3] * input_curr[threadIdx.y - 1][threadIdx.x]
                                        + coefficients[4] * input_curr[threadIdx.y + 1][threadIdx.x]
                                        + coefficients[5] * input_prev_float
                                        + coefficients[6] * input_next_float;
        __syncthreads();

        input_prev_float = input_curr[threadIdx.y][threadIdx.x];
        input_curr[threadIdx.y][threadIdx.x] = input_next_float;
    }
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
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
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