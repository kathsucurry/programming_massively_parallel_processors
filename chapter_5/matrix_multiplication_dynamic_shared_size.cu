#include <stdio.h>
#include <cuda_runtime.h>

// BLOCK_SIZE and TILE_WIDTH should be identical.
#define BLOCK_SIZE 4
#define TILE_WIDTH 4


/**
 *  Performs a simple matrix multiplication with shared memory allowing varying shared memory size.
 * 
 *  Note that shared memory M and shared memory N have identical size.
 */
__global__
void MatrixMultiplicationKernel(
    float* matrix_M,
    float* matrix_N,
    float* matrix_Out,
    int Width,
    unsigned shared_M_N_size
) {
    // Initialize the shared memories.
    extern __shared__ float shared_M_N[];

    float *shared_M = (float *) shared_M_N;
    float *shared_N = (float *) shared_M_N + shared_M_N_size / 2;

    int block_x = blockIdx.x;
    int block_y = blockIdx.y;
    int thread_x = threadIdx.x;
    int thread_y = threadIdx.y;

    // Identify the row and column of the matrix_Out matrix to work on.
    // Note that here, block dimension = TILE_WIDTH.
    int Row = block_y * TILE_WIDTH + thread_y;
    int Col = block_x * TILE_WIDTH + thread_x;

    // Loop over the tiles required to compute matrix_Out elements.
    float out_value = 0;
    for (int phase = 0; phase < ceil(Width*1.0/TILE_WIDTH); ++phase) {
        // Collaboratively load M and N tiles into shared memory.
        int in_matrix_M_index = Row*Width + phase*TILE_WIDTH + thread_x;
        int in_matrix_N_index = (phase*TILE_WIDTH + thread_y)*Width + Col;

        // Check for boundary for both dimensions.
        if (Row < Width && phase*TILE_WIDTH + thread_x < Width)
            shared_M[thread_y*TILE_WIDTH + thread_x] = matrix_M[in_matrix_M_index];
        else
            shared_M[thread_y*TILE_WIDTH + thread_x] = 0;

        if (phase*TILE_WIDTH + thread_y < Width && Col < Width)
            shared_N[thread_y*TILE_WIDTH + thread_x] = matrix_N[in_matrix_N_index];
        else
            shared_N[thread_y*TILE_WIDTH + thread_x] = 0;
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            out_value += shared_M[thread_y*TILE_WIDTH + k] * shared_N[k*TILE_WIDTH + thread_x];
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
    int Width,
    unsigned total_shared_memory_size
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
    MatrixMultiplicationKernel<<<dimGrid, dimBlock, total_shared_memory_size>>>(matrix_M_d, matrix_N_d, matrix_Out_d, Width, total_shared_memory_size);

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
    
    // For simplicity, we define the shared memory size here instead of
    // it being adjusted based on the device model.
    unsigned shared_memory_size = sizeof(float) * 4 * 2;

    // Define identical matrices M and N where each element = 1 .. Width * Width.
    float * matrix_M = (float *) malloc(Width * Width * sizeof(float));
    float * matrix_N = (float *) malloc(Width * Width * sizeof(float));
    float * matrix_Out = (float *) malloc(Width * Width * sizeof(float));

    for (int i = 1; i <= Width * Width; ++i) {
        matrix_M[i - 1] = i;
        matrix_N[i - 1] = i;
    }

    runMatrixMultiplication(matrix_M, matrix_N, matrix_Out, Width, shared_memory_size);

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