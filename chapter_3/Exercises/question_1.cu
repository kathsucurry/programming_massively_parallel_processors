/**
 * Question:
 * Implement a matrix multiplication kernel on same-width square matrices where:
 * a. each thread produce one output matrix row.
 * b. each thread produce one output matrix column.
 * 
 */

#include <stdio.h>
#include <cuda_runtime.h>

// Thread block size.
#define BLOCK_SIZE 2

typedef void (MatrixMultiplicationFuction)(float*, float*, float*, int);


__global__
void Question1AKernel(
    float* matrix_M,
    float* matrix_N,
    float* matrix_Out,
    int Width
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < Width) {
        for (int col = 0; col < Width; ++col) {
            float out_value = 0;
            for (int k = 0; k < Width; ++k) {
                out_value += matrix_M[row * Width + k] * matrix_N[Width * k + col];
            }
            matrix_Out[row * Width + col] = out_value;
        }
    }
}


__global__
void Question1BKernel(
    float* matrix_M,
    float* matrix_N,
    float* matrix_Out,
    int Width
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < Width) {
        for (int row = 0; row < Width; ++row) {
            float out_value = 0;
            for (int k = 0; k < Width; ++k) {
                out_value += matrix_M[row * Width + k] * matrix_N[Width * k + col];
            }
            matrix_Out[row * Width + col] = out_value;
        }
    }
}


void runMatrixMultiplication(
    float* matrix_M_h,
    float* matrix_N_h,
    float* matrix_Out_h,
    int Width,
    MatrixMultiplicationFuction* matmul_kernel
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
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(ceil(Width / (BLOCK_SIZE * 0.1)));
    matmul_kernel<<<dimGrid, dimBlock>>>(matrix_M_d, matrix_N_d, matrix_Out_d, Width);

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

    // Pick either question 1a or question 1b kernel for testing purposes.
    MatrixMultiplicationFuction* kernel_function = Question1BKernel;
    runMatrixMultiplication(matrix_M, matrix_N, matrix_Out, Width, kernel_function);

    for (int i = 0; i < Width; ++i) {
        for (int j = 0; j < Width; ++j) {
            printf("%.2f ", matrix_Out[i * Width + j]);
        }
        printf("\n");
    }

    return 0;
}