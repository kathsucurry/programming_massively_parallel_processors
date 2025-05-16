#include <stdio.h>
#include <cuda_runtime.h>

// BLOCK_SIZE and TILE_WIDTH should be identical.
#define BLOCK_SIZE 2
#define TILE_WIDTH 2


/**
 *  Performs a matrix multiplication process without corner turning (fig. 6.4).
 */
__global__
void NoCornerTurningMatrixMultiplicationKernel(
    float* matrix_M,
    float* matrix_N,
    float* matrix_Out,
    int Width
) {
    // Initialize the shared memories.
    __shared__  float shared_M[TILE_WIDTH][TILE_WIDTH];
    __shared__  float shared_N[TILE_WIDTH][TILE_WIDTH];

    int block_x = blockIdx.x;
    int block_y = blockIdx.y;
    int thread_x = threadIdx.x;
    int thread_y = threadIdx.y;

    // Identify the row and column of the matrix_Out matrix to work on.
    // Note that here, block dimension = TILE_WIDTH.
    int Row = block_y * blockDim.y + thread_y;
    int Col = block_x * blockDim.x + thread_x;

    // Loop over the tiles required to compute matrix_Out elements.
    float out_value = 0;
    for (int phase = 0; phase < ceil(Width*1.0/TILE_WIDTH); ++phase) {
        // Collaboratively load M and N tiles into shared memory.
        int in_matrix_M_index = Row*Width + phase*TILE_WIDTH + thread_x;
        int in_matrix_N_index = Col*Width + phase*TILE_WIDTH + thread_y;

        // Check for boundary for both dimensions.
        if (Row < Width && phase*TILE_WIDTH + thread_x < Width)
            shared_M[thread_y][thread_x] = matrix_M[in_matrix_M_index];
        else
            shared_M[thread_y][thread_x] = 0;

        if (Col < Width && phase*TILE_WIDTH + thread_y < Width)
            shared_N[thread_y][thread_x] = matrix_N[in_matrix_N_index];
        else
            shared_N[thread_y][thread_x] = 0;
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            out_value += shared_M[thread_y][k] * shared_N[k][thread_x];
        }
        __syncthreads();
    }

    if (Row < Width && Col < Width)
        matrix_Out[Row * Width + Col] = out_value;
}


/**
 *  Performs a matrix multiplication process utilizing corner turning (fig. 6.4).
 */
__global__
void CornerTurningMatrixMultiplicationKernel(
    float* matrix_M,
    float* matrix_N,
    float* matrix_Out,
    int Width
) {
    // Initialize the shared memories.
    __shared__  float shared_M[TILE_WIDTH][TILE_WIDTH];
    __shared__  float shared_N[TILE_WIDTH][TILE_WIDTH];

    int block_x = blockIdx.x;
    int block_y = blockIdx.y;
    int thread_x = threadIdx.x;
    int thread_y = threadIdx.y;

    // Identify the row and column of the matrix_Out matrix to work on.
    // Note that here, block dimension = TILE_WIDTH.
    int Row = block_y * blockDim.y + thread_y;
    int Col = block_x * blockDim.x + thread_x;

    // Loop over the tiles required to compute matrix_Out elements.
    float out_value = 0;
    for (int phase = 0; phase < ceil(Width*1.0/TILE_WIDTH); ++phase) {
        // Collaboratively load M and N tiles into shared memory.
        int in_matrix_M_index = Row*Width + phase*TILE_WIDTH + thread_x;

        // To perform corner turning, we want to ensure that consecutive threads load
        // consecutive elements. Recall that <block_x * blockDim.x> indicates the start of a new block
        // in the 1st dimension.
        int in_matrix_N_index = (block_x * blockDim.x + thread_y)*Width + phase*TILE_WIDTH + thread_x;

        // Check for boundary for both dimensions.
        if (Row < Width && phase*TILE_WIDTH + thread_x < Width)
            shared_M[thread_y][thread_x] = matrix_M[in_matrix_M_index];
        else
            shared_M[thread_y][thread_x] = 0;

        if (block_x * blockDim.x + thread_y < Width && phase*TILE_WIDTH + thread_x < Width)
            // In order to keep the loop implementation the same, switch the 2 axes
            // when storing shared_N.
            shared_N[thread_x][thread_y] = matrix_N[in_matrix_N_index];
        else
            shared_N[thread_x][thread_y] = 0;
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            out_value += shared_M[thread_y][k] * shared_N[k][thread_x];
        }
        __syncthreads();
    }

    if (Row < Width && Col < Width)
        matrix_Out[Row * Width + Col] = out_value;
}


void runMatrixMultiplication(
    float* matrix_M_h,
    float* matrix_N_h,
    float* matrix_Out_h,
    int Width
) {
    // Get size in bytes.
    size_t size = Width * Width * sizeof(float);

    // Load and copy matrix M and N to device memory.
    float * matrix_M_d, * matrix_N_d, * matrix_Out_d;
    cudaMalloc((void***)&matrix_M_d, size);
    cudaMemcpy(matrix_M_d, matrix_M_h, size, cudaMemcpyHostToDevice);

    cudaMalloc((void***)&matrix_N_d, size);
    cudaMemcpy(matrix_N_d, matrix_N_h, size, cudaMemcpyHostToDevice);

    cudaMalloc((void***)&matrix_Out_d, size);

    // Invoke kernel.
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(ceil(Width / (BLOCK_SIZE * 1.0)), ceil(Width / (BLOCK_SIZE * 1.0)));
    CornerTurningMatrixMultiplicationKernel<<<dimGrid, dimBlock>>>(matrix_M_d, matrix_N_d, matrix_Out_d, Width);

    // Copy the output matrix from the device memory.
    cudaMemcpy(matrix_Out_h, matrix_Out_d, size, cudaMemcpyDeviceToHost);

    // Free device vectors.
    cudaFree(matrix_M_d);
    cudaFree(matrix_N_d);
    cudaFree(matrix_Out_d);
}


int main() {
    int Width = 9;

    // Define identical matrices M and N where each element = 1 .. Width * Width.
    float * matrix_M = (float *) malloc(Width * Width * sizeof(float));
    float * matrix_N = (float *) malloc(Width * Width * sizeof(float));
    float * matrix_Out = (float *) malloc(Width * Width * sizeof(float));

    // Variables used to store matrix N in column-major order.
    int current_original_row = 0;
    int current_original_col = 0;

    for (int i = 1; i <= Width * Width; ++i) {
        // Matrix M is stored in row-major order.
        matrix_M[i - 1] = i;
        
        // Matrix N is stored in column-major order.
        int row = (i - 1) / Width;
        if (row != current_original_row) {
            current_original_row = row;
            current_original_col = 0;
        }
        matrix_N[row + current_original_col * Width] = i;
        current_original_col += 1;
    }

    runMatrixMultiplication(matrix_M, matrix_N, matrix_Out, Width);

    for (int i = 0; i < Width; ++i) {
        for (int j = 0; j < Width; ++j) {
            printf("%.2f ", matrix_Out[i * Width + j]);
        }
        printf("\n");
    }

    free(matrix_M);
    free(matrix_N);
    free(matrix_Out);

    return 0;
}