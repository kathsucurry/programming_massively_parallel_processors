#include <stdio.h>
#include <cuda_runtime.h>
 
// Thread block size.
#define BLOCK_SIZE 16
 
/**
 * The matrix-vector multiplication: vector A = matrix B (*) vector C where
 *      A[i] = sum_j B[i][j] * C[j] (dot product).
 * Assume square matrices for simplicity.
 *
 * Note that matrix_B would be a 1D array with length Width x Width.
 *
 */
__global__
void MatrixVectorMultiplicationKernel(
    float * vector_A,
    float * matrix_B,
    float * vector_C,
    int Width
) {
    int out_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_index < Width) {
        float sum_value = 0;
        for (int j = 0; j < Width; ++j) {
            sum_value += matrix_B[out_index * Width + j] * vector_C[j];
        }
        vector_A[out_index] = sum_value;
    }
}
 
/**
 * The host function, to deal with memory allocations and kernel function calls.
 */
void runMatrixVectorMultiplication(
    float * vector_A_h,
    float * matrix_B_h,
    float * vector_C_h,
    int Width
) {
    // Get size in bytes.
    size_t matrix_size = Width * Width * sizeof(float);
    size_t vector_size = Width * sizeof(float);

    // Load and copy matrix B and vector C to device memory.
    float * vector_A_d, * matrix_B_d, * vector_C_d;
    cudaMalloc((void***)&matrix_B_d, matrix_size);
    cudaMemcpy(matrix_B_d, matrix_B_h, matrix_size, cudaMemcpyHostToDevice);

    cudaMalloc((void***)&vector_C_d, vector_size);
    cudaMemcpy(vector_C_d, vector_C_h, vector_size, cudaMemcpyHostToDevice);
 
    cudaMalloc((void***)&vector_A_d, vector_size);
 
    // Invoke kernel.
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(ceil(Width / (BLOCK_SIZE * 1.0)));
 
    MatrixVectorMultiplicationKernel<<<dimGrid, dimBlock>>>(vector_A_d, matrix_B_d, vector_C_d, Width);

    // Copy the output matrix from the device memory.
    cudaMemcpy(vector_A_h, vector_A_d, vector_size, cudaMemcpyDeviceToHost);

    // Free device vectors.
    cudaFree(vector_A_d);
    cudaFree(matrix_B_d);
    cudaFree(vector_C_d);
}


int main() {
    // Matrices are to be stored in row-major order.
    int Width = 4;
    float * matrix_B = (float *) malloc(Width * Width * sizeof(float));
    for (int i = 0; i < Width * Width; ++i) {
        matrix_B[i] = i * 1.0;
    }

    float * vector_C = (float *) malloc(Width * sizeof(float));
    for (int i = 0; i < Width; ++i) {
        vector_C[i] = i * 1.0;
    }

    float * vector_A = (float *) malloc(Width * sizeof(float));

    runMatrixVectorMultiplication(vector_A, matrix_B, vector_C, Width);

    // Should output 14, 38, 62, 86.
    for (int i = 0; i < Width; ++i) {
        printf("%.2f\n", vector_A[i]);
    }

    free(matrix_B);
    free(vector_C);
    free(vector_A);
    return 0;
}