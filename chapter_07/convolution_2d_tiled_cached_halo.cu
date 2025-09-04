#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 4
#define FILTER_RADIUS 1

#define TILE_DIM BLOCK_SIZE

// Declare the filter array that utilizes constant memory.
__constant__ float CONST_FILTER[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];


/**
 *  Perform tiled 2D convolution with caching for halos; corresponds to figure 7.15.
 */
__global__
void Convolution2DTiledKernelWithHaloCaching(
    float* input_array,
    float* output_array,
    int width,
    int height
) {
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    
    // Load input tile.
    __shared__ float shared_array[TILE_DIM][TILE_DIM];
    // No need to check if row/col >=0 since they will always be positive.
    if (row < height && col < width)
        shared_array[threadIdx.y][threadIdx.x] = input_array[row * width + col];
    else
        shared_array[threadIdx.y][threadIdx.x] = 0.0f;
    
    __syncthreads();

    // Calculate output elements.
    // Turn off the threads at the edges of the block.
    if (!(col < width && row < height))
        return;

    float out_value = 0.0f;
    for (int conv_row = -FILTER_RADIUS; conv_row <= FILTER_RADIUS; ++conv_row) {
        for (int conv_col = -FILTER_RADIUS; conv_col <= FILTER_RADIUS; ++conv_col) {
            // Obtain the cell index of the filter.
            int filter_row = conv_row + FILTER_RADIUS;
            int filter_col = conv_col + FILTER_RADIUS;

            // Check for halo cells.
            if (threadIdx.x + conv_col >= 0 &&
                threadIdx.x + conv_col < TILE_DIM &&
                threadIdx.y + conv_row >= 0 &&
                threadIdx.y + conv_row < TILE_DIM) {
                    out_value += CONST_FILTER[filter_row][filter_col] * shared_array[threadIdx.y + conv_row][threadIdx.x + conv_col];
            } else {
                // Check for boundary edges.
                if (row + conv_row >= 0 &&
                    row + conv_row < height &&
                    col + conv_col >=0 &&
                    col + conv_col < width) {
                        out_value += CONST_FILTER[filter_row][filter_col] * input_array[(row + conv_row)* width + col + conv_col];
                }
            }
        }
    }
    output_array[row * width + col] = out_value;
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
    dim3 dimGrid(ceil(width / (TILE_DIM * 1.0)), ceil(height / (TILE_DIM * 1.0)));
    Convolution2DTiledKernelWithHaloCaching<<<dimGrid, dimBlock>>>(
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