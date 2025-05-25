#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 4
#define FILTER_RADIUS 1

#define IN_TILE_DIM BLOCK_SIZE
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2*(FILTER_RADIUS))

// Declare the filter array that utilizes constant memory.
__constant__ float CONST_FILTER[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];


/**
 *  Perform tiled 2D convolution; corresponds to figure 7.12.
 */
__global__
void Convolution2DTiledKernel(
    float* input_array,
    float* output_array,
    int width,
    int height
) {
    int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
    int row = blockIdx.y * OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;
    
    // Load input tile.
    __shared__ float shared_array[IN_TILE_DIM][IN_TILE_DIM];
    if (row >=0 && row < height && col >=0 && col < width)
        shared_array[threadIdx.y][threadIdx.x] = input_array[row * width + col];
    else
        shared_array[threadIdx.y][threadIdx.x] = 0.0f;
    
    __syncthreads();

    // Calculate tile offset.
    int tile_col = threadIdx.x - FILTER_RADIUS;
    int tile_row = threadIdx.y - FILTER_RADIUS;

    // Turn off the threads at the edges of the block.
    if (!(col >=0 && col < width && row >= 0 and row < height))
        return;
    
    if (tile_col >= 0 && tile_col < OUT_TILE_DIM && tile_row >= 0 && tile_row < OUT_TILE_DIM) {
        float out_value = 0.0f;
        for (int conv_row = -FILTER_RADIUS; conv_row < FILTER_RADIUS + 1; ++conv_row) {
            for (int conv_col = -FILTER_RADIUS; conv_col < FILTER_RADIUS + 1; ++conv_col) {
                int filter_row = conv_row + FILTER_RADIUS;
                int filter_col = conv_col + FILTER_RADIUS;
                out_value += CONST_FILTER[filter_row][filter_col] * shared_array[tile_row+filter_row][tile_col+filter_col];
            }
        }
        output_array[row * width + col] = out_value;
    }
}


void runConvolution2D(
    float* input_array_h,
    float* filter_h,
    float* output_array_h,
    int width,
    int height
) {
    // Get size in bytes.
    size_t size_input = width * height * sizeof(float);
    size_t size_filter = (2*FILTER_RADIUS+1) * (2*FILTER_RADIUS+1) * sizeof(float);

    // Load and copy input_array and filter to device memory.
    float * input_array_d, * output_array_d;
    
    cudaMalloc((void***)&input_array_d, size_input);
    cudaMemcpy(input_array_d, input_array_h, size_input, cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(CONST_FILTER, filter_h, size_filter);

    cudaMalloc((void***)&output_array_d, size_input);

    // Invoke kernel.
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(ceil(width / (OUT_TILE_DIM * 1.0)), ceil(height / (OUT_TILE_DIM * 1.0)));
    Convolution2DTiledKernel<<<dimGrid, dimBlock>>>(
        input_array_d,
        output_array_d,
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
    // Define input array as a 6 x 5 array with values 1 .. 30.
    int height = 6, width = 5;
    float input_array[30];
    for (int i = 0; i < height * width; ++i)
        input_array[i] = i + 1;

    // Define the filter as a 3 x 3 array with values 1 .. 9.
    float filter[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    float output_array[12];

    runConvolution2D(
        input_array,
        filter,
        output_array,
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