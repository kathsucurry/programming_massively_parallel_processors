#include <stdio.h>
#include <cuda_runtime.h>

// BLOCK_SIZE and TILE_WIDTH should be identical.
#define BLOCK_SIZE 8
#define TILE_WIDTH 8
#define COARSE_FACTOR 4


/**
 *  Performs a simple matrix multiplication with thread coarsening.
 * 
 *  Note that shared memory M and shared memory N have identical size.
 */
__global__
void MatrixMultiplicationKernel(
    float* matrix_M,
    float* matrix_N,
    float* matrix_Out,
    int Width
) {
    __shared__ float shared_M[TILE_WIDTH][TILE_WIDTH];
    __shared__ float shared_N[TILE_WIDTH][TILE_WIDTH];

    int block_x = blockIdx.x;
    int block_y = blockIdx.y;
    int thread_x = threadIdx.x;
    int thread_y = threadIdx.y;

    // Identify the row and column of the output element to work on.
    int Row = block_y * TILE_WIDTH + thread_y;
    int ColStart = block_x * TILE_WIDTH*COARSE_FACTOR + thread_x;

    // Initialize values for all output elements.
    float out_values[COARSE_FACTOR];
    for (int c = 0; c < COARSE_FACTOR; ++c) {
        out_values[c] = 0.0f;
    }

    // Loop over the tiles required to compute out elements.
    for (int phase = 0; phase < ceil(Width*1.0/TILE_WIDTH); ++phase) {
        int in_matrix_M_index = Row*Width + phase*TILE_WIDTH + thread_x;

        // Collaborative loading of M tile into shared memory.
        if (Row < Width && phase*TILE_WIDTH + thread_x < Width)
            shared_M[thread_y][thread_x] = matrix_M[in_matrix_M_index];
        else
            shared_M[thread_y][thread_x] = 0;

        for (int c = 0; c < COARSE_FACTOR; ++c) {
            int Col = ColStart + c*TILE_WIDTH;
            int in_matrix_N_index = (phase*TILE_WIDTH + thread_y)*Width + Col;

            // Collaborative loading of N tile into shared memory.
            if (phase*TILE_WIDTH + thread_y < Width && Col < Width)
                shared_N[thread_y][thread_x] = matrix_N[in_matrix_N_index];
            else
                shared_N[thread_y][thread_x] = 0;
            __syncthreads();

            for (int k = 0; k < TILE_WIDTH; ++k) {
                out_values[c] += shared_M[thread_y][k] * shared_N[k][thread_x];
            }
            __syncthreads();
        }
    }

    for (int c = 0; c < COARSE_FACTOR; ++c) {
        int Col = ColStart + c*TILE_WIDTH;
        if (Row < Width && Col < Width)
            matrix_Out[Row * Width + Col] = out_values[c];
    }
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
    dim3 dimGrid(ceil(Width * 1.0 / BLOCK_SIZE / COARSE_FACTOR), ceil(Width * 1.0 / BLOCK_SIZE));
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
    int Width = 20;
    
    // Define identical matrices M and N where each element = 1 .. Width * Width.
    float * matrix_M = (float *) malloc(Width * Width * sizeof(float));
    float * matrix_N = (float *) malloc(Width * Width * sizeof(float));
    float * matrix_Out = (float *) malloc(Width * Width * sizeof(float));

    for (int i = 1; i <= Width * Width; ++i) {
        matrix_M[i - 1] = i % 7;
        matrix_N[i - 1] = i % 7;
    }

    runMatrixMultiplication(matrix_M, matrix_N, matrix_Out, Width);

    for (int i = 0; i < Width; ++i) {
        for (int j = 0; j < Width; ++j) {
            printf("%.0f ", matrix_Out[i * Width + j]);
        }
        printf("\n");
    }

    free(matrix_M);
    free(matrix_N);
    free(matrix_Out);

    return 0;
}