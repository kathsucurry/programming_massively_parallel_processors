#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 4
#define FILTER_RADIUS 1

#define IN_TILE_DIM BLOCK_SIZE
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2*(FILTER_RADIUS))

// Declare the filter array that utilizes constant memory.
__constant__ float CONST_FILTER[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];


/**
 *  Perform tiled 2D convolution; corresponds to figure 7.12.
 */
__global__
void Convolution3DTiledKernel(
    float* input_array,
    float* output_array,
    int dim_x_size,
    int dim_y_size,
    int dim_z_size
) {
    int out_x_index = blockIdx.x * OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
    int out_y_index = blockIdx.y * OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;
    int out_z_index = blockIdx.z * OUT_TILE_DIM + threadIdx.z - FILTER_RADIUS;
    int out_index = out_z_index * dim_x_size * dim_y_size + out_y_index * dim_x_size + out_x_index;

    // Load input tile.
    __shared__ float shared_array[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];

    if (out_x_index >= 0 && out_x_index < dim_x_size &&
        out_y_index >= 0 && out_y_index < dim_y_size &&
        out_z_index >= 0 && out_z_index < dim_z_size)
        shared_array[threadIdx.z][threadIdx.y][threadIdx.x] = input_array[out_index];
    else
        shared_array[threadIdx.z][threadIdx.y][threadIdx.x] = 0.0f;
    
    __syncthreads();

    // Calculate tile offset.
    int tile_x = threadIdx.x - FILTER_RADIUS;
    int tile_y = threadIdx.y - FILTER_RADIUS;
    int tile_z = threadIdx.z - FILTER_RADIUS;

    // Turn off the threads at the edges of the block.
    if (!(out_x_index >= 0 && out_x_index < dim_x_size &&
            out_y_index >= 0 && out_y_index < dim_y_size &&
            out_z_index >= 0 && out_z_index < dim_z_size))
        return;
    
    if (tile_x >= 0 && tile_x < OUT_TILE_DIM && tile_y >= 0 && tile_y < OUT_TILE_DIM && tile_z >= 0 && tile_z < OUT_TILE_DIM) {
        float out_value = 0.0f;
        for (int conv_z = -FILTER_RADIUS; conv_z < FILTER_RADIUS + 1; ++conv_z) {
            for (int conv_y = -FILTER_RADIUS; conv_y < FILTER_RADIUS + 1; ++conv_y) {
                for (int conv_x = -FILTER_RADIUS; conv_x < FILTER_RADIUS + 1; ++conv_x) {
                    int filter_z = conv_z + FILTER_RADIUS;
                    int filter_y = conv_y + FILTER_RADIUS;
                    int filter_x = conv_x + FILTER_RADIUS;
                    out_value += CONST_FILTER[filter_z][filter_y][filter_x] * shared_array[tile_z+filter_z][tile_y+filter_y][tile_x+filter_x];
                }
            }
        }
        output_array[out_index] = out_value;
    }
}


void runConvolution3D(
    float* input_array_h,
    float* filter_h,
    float* output_array_h,
    int dim_x_size,
    int dim_y_size,
    int dim_z_size
) {
    // Get size in bytes.
    size_t size_input = dim_x_size * dim_y_size * dim_z_size * sizeof(float);
    size_t size_filter = (2*FILTER_RADIUS+1) * (2*FILTER_RADIUS+1) * (2*FILTER_RADIUS+1) * sizeof(float);

    // Load and copy input_array and filter to device memory.
    float * input_array_d, * output_array_d;
    
    cudaMalloc((void***)&input_array_d, size_input);
    cudaMemcpy(input_array_d, input_array_h, size_input, cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(CONST_FILTER, filter_h, size_filter);

    cudaMalloc((void***)&output_array_d, size_input);

    // Invoke kernel.
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(
        ceil(dim_x_size / (OUT_TILE_DIM * 1.0)),
        ceil(dim_y_size / (OUT_TILE_DIM * 1.0)),
        ceil(dim_z_size / (OUT_TILE_DIM * 1.0)));
    Convolution3DTiledKernel<<<dimGrid, dimBlock>>>(
        input_array_d,
        output_array_d,
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
    // Define input array as a 8 x 7 x 6 array with input (1 .. 336) % 10.
    int dim_x_size = 6, dim_y_size = 7, dim_z_size = 8;
    float input_array[dim_x_size * dim_y_size * dim_z_size];

    for (int i = 1; i < dim_x_size * dim_y_size * dim_z_size + 1; ++i) 
        input_array[i-1] = i % 10;

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