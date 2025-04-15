/**
 * Question:
 * Implement a matrix multiplication kernel on same-width square matrices where:
 * a. each thread produces one output matrix row.
 * b. each thread produces one output matrix column.
 * 
 * Please refer to the README.md in the Exercises directory for
 * further descriptions of what I am trying to do in the code.
 */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
 
// Thread block size.
#define BLOCK_SIZE 16
 
typedef void (MatrixMultiplicationFuction)(int*, int*, int*, int);
 
 
/**
 * The matmul kernel function where each thread produces one output matrix row.
 */
__global__
void Question1AKernel(
    int* matrix_M,
    int* matrix_N,
    int* matrix_Out,
    int Width
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
 
    if (row < Width) {
        for (int col = 0; col < Width; ++col) {
            int out_value = 0;
            for (int k = 0; k < Width; ++k) {
                out_value += matrix_M[row * Width + k] * matrix_N[Width * k + col];
            }
            matrix_Out[row * Width + col] = out_value;
        }
    }
}
 
 
/**
 * The matmul kernel function where each thread produces one output column row.
 */
__global__
void Question1BKernel(
    int* matrix_M,
    int* matrix_N,
    int* matrix_Out,
    int Width
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
 
    if (col < Width) {
        for (int row = 0; row < Width; ++row) {
            int out_value = 0;
            for (int k = 0; k < Width; ++k) {
                out_value += matrix_M[row * Width + k] * matrix_N[Width * k + col];
            }
            matrix_Out[row * Width + col] = out_value;
        }
    }
}
 
 
/**
 * The host function, to deal with memory allocations and kernel function calls.
 */
void runMatrixMultiplication(
    int* matrix_M_h,
    int* matrix_N_h,
    int* matrix_Out_h,
    int Width,
    MatrixMultiplicationFuction* matmul_kernel
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
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(ceil(Width / (BLOCK_SIZE * 0.1)));
 
    // Initialize CUDA events to time the kernel function run.
    // For more information: https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul_kernel<<<dimGrid, dimBlock>>>(matrix_M_d, matrix_N_d, matrix_Out_d, Width);
    cudaEventRecord(stop);

    // Copy the output matrix from the device memory.
    cudaMemcpy(matrix_Out_h, matrix_Out_d, size, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float time_spent_ms = 0;
    cudaEventElapsedTime(&time_spent_ms, start, stop);
    printf("Total duration: %.7f\n", time_spent_ms / 1000.0);

    // Free device vectors.
    cudaFree(matrix_M_d);
    cudaFree(matrix_N_d);
    cudaFree(matrix_Out_d);
}


void run_kernel(
    int * matrix_M,
    int * matrix_N,
    int Width,
    MatrixMultiplicationFuction* matmul_kernel,
    const char * label
) {
    printf("Running %s...\n", label);

    int * matrix_Out = (int *) malloc(Width * Width * sizeof(int));

    runMatrixMultiplication(matrix_M, matrix_N, matrix_Out, Width, matmul_kernel);
    
    free(matrix_Out);
    return;
}
