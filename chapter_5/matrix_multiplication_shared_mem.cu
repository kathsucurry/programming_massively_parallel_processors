#include <stdio.h>
#include <cuda_runtime.h>

// BLOCK_SIZE and TILE_WIDTH should be identical.
#define BLOCK_SIZE 2
#define TILE_WIDTH 2


/**
 *  Performs a simple matrix multiplication with shared memory, assuming all matrices are square with the same Width.
 * 
 *  One example code can be found here: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory.
 */
__global__
void MatrixMultiplicationKernel(
    int* matrix_M,
    int* matrix_N,
    int* matrix_Out,
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
    int out_value = 0;
    for (int phase = 0; phase < ceil(Width*1.0/TILE_WIDTH); ++phase) {
        // Collaboratively load M and N tiles into shared memory.
        int in_matrix_M_index = Row*Width + phase*TILE_WIDTH + thread_x;
        int in_matrix_N_index = (phase*TILE_WIDTH + thread_y)*Width + Col;

        // Check for boundary for both dimensions.
        if (Row < Width && phase*TILE_WIDTH + thread_x < Width)
            shared_M[thread_y][thread_x] = matrix_M[in_matrix_M_index];
        else
            shared_M[thread_y][thread_x] = 0;

        if (phase*TILE_WIDTH + thread_y < Width && Col < Width)
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


void runMatrixMultiplication(
    int* matrix_M_h,
    int* matrix_N_h,
    int* matrix_Out_h,
    int Width
) {
    // Get size in bytes.
    size_t size = Width * Width * sizeof(int);

    // Load and copy matrix M and N to device memory.
    int * matrix_M_d, * matrix_N_d, * matrix_Out_d;
    cudaMalloc((void***)&matrix_M_d, size);
    cudaMemcpy(matrix_M_d, matrix_M_h, size, cudaMemcpyHostToDevice);

    cudaMalloc((void***)&matrix_N_d, size);
    cudaMemcpy(matrix_N_d, matrix_N_h, size, cudaMemcpyHostToDevice);

    cudaMalloc((void***)&matrix_Out_d, size);

    // Invoke kernel.
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(ceil(Width / (BLOCK_SIZE * 1.0)), ceil(Width / (BLOCK_SIZE * 1.0)));
    MatrixMultiplicationKernel<<<dimGrid, dimBlock>>>(matrix_M_d, matrix_N_d, matrix_Out_d, Width);

    // Copy the output matrix from the device memory.
    cudaMemcpy(matrix_Out_h, matrix_Out_d, size, cudaMemcpyDeviceToHost);

    // Free device vectors.
    cudaFree(matrix_M_d);
    cudaFree(matrix_N_d);
    cudaFree(matrix_Out_d);
}


int main() {
    // Matrices are stored in row-major order.
    int Width = 9;

    // Define identical matrices M and N where each element = 1 .. Width * Width.
    int * matrix_M = (int *) malloc(Width * Width * sizeof(int));
    int * matrix_N = (int *) malloc(Width * Width * sizeof(int));
    int * matrix_Out = (int *) malloc(Width * Width * sizeof(int));

    for (int i = 1; i <= Width * Width; ++i) {
        matrix_M[i - 1] = i;
        matrix_N[i - 1] = i;
    }

    runMatrixMultiplication(matrix_M, matrix_N, matrix_Out, Width);

    for (int i = 0; i < Width; ++i) {
        for (int j = 0; j < Width; ++j) {
            printf("%d ", matrix_Out[i * Width + j]);
        }
        printf("\n");
    }

    free(matrix_M);
    free(matrix_N);
    free(matrix_Out);

    return 0;
}