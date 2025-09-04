#include <stdio.h>

#include <cuda_runtime.h>


__global__ void vecAddKernel(float* A, float* B, float* C, int n) {
    // Each thread performs one pair-wise addition; all threads execute the
    // same kernel code.
    // __global__ indicates a kernel function.

    // The variable i is a local variable, private to each thread.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}


void vectorAdd(float* A_h, float* B_h, float* C_h, int n) {
    // Get size in bytes.
    int size = n * sizeof(float);
    float *A_d, *B_d, *C_d;

    // Part 1: Allocate device memory for A, B, and C.
    // A_d would then point to the device global memory region allocated
    // for the A vector.
    cudaMalloc((void***)&A_d, size);
    cudaMalloc((void***)&B_d, size);
    cudaMalloc((void***)&C_d, size);

    // Copy A and B to device memory.
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    // Part 2: Call kernel to launch a grid of threads.
    // to perform the actual vector addition.
    // ceil(n/256.0) --> the number of blocks in the grid.
    // 256 --> the number of threads in each block.
    vecAddKernel<<<ceil(n/256.0), 256>>>(A_d, B_d, C_d, n);

    // Part 3: copy C from the device memory.
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    // Free device vectors.
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

}


int main() {
    // A very simple example of vectors with only length of 4.
    float A[] = {1, 2, 3, 4};
    float B[] = {2, 3, 4, 5};
    float C[4];
    int N = 4;

    // Memory allocation for arrays A, B, and C.
    // I/O to read A and B, N elements each.
    vectorAdd(A, B, C, N);
    for (int i = 0; i < N; ++i) {
        printf("%.2f ", C[i]);
    }
}