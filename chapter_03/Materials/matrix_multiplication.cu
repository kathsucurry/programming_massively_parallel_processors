#include <stdio.h>
#include <cuda_runtime.h>

// Thread block size.
#define BLOCK_SIZE 16


/**
 *  Performs a simple matrix multiplication, assuming all matrices are square with the same Width.
 * 
 *  One example code can be found here: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory.
 */
__global__
void MatrixMultiplicationKernel(
    float* matrix_M,
    float* matrix_N,
    float* matrix_Out,
    int Width
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((row < Width) && (col < Width)) {
        float out_value = 0;
        for (int k = 0; k < Width; ++k) {
            out_value += matrix_M[row * Width + k] * matrix_N[Width * k + col];
        }
        matrix_Out[row * Width + col] = out_value;
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
    int Width = 3;
    // A very simple example.
    float matrix_M[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    float matrix_N[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    float matrix_Out[Width * Width];

    runMatrixMultiplication(matrix_M, matrix_N, matrix_Out, Width);

    for (int i = 0; i < Width; ++i) {
        for (int j = 0; j < Width; ++j) {
            printf("%.2f ", matrix_Out[i * Width + j]);
        }
        printf("\n");
    }

    return 0;
}