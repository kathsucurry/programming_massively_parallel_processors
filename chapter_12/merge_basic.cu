/**
 *  Perform basic merge kernel, corresponds to Fig. 12.9.
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define THREADS_NUM_PER_BLOCK 4


/**
 * Retrieve the co-rank using binary search, i.e., the beginning positions of A and B that will be
 * merged into C that is assigned to the thread.
 * 
 * @param k The rank of the C element of interest, i.e., the start of the output index.
 * @param A The subarray A.
 * @param m The length of subarray A.
 * @param B The subarray B.
 * @param n The length of subarray B.
 * 
 */
__device__
int ObtainCoRank(int k, int* A, int m, int* B, int n) {
    // Initialize the co-rank values, i.e., the highest possible values.
    int i = k < m ? k : m; // i = min(k, m)
    int j = k - i;

    // Initialize the lowest possible co-rank values.
    int i_low = 0 > (k-n) ? 0 : k-n; // i_low = max(0, k-n)
    int j_low = 0 > (k-m) ? 0 : k-m; // j_low = max(0, k-m)

    int delta;

    // The goal is to find a pair of i and j such that A[i-1] <= B[j]
    // and B[j-1] < A[i].
    while (true) {
        // Decrease i and increase j if i is too high.
        // Both i and j are changed to maintain the property where i + j = k.
        if (i > 0 && j < n && A[i-1] > B[j]) {
            delta = ((i - i_low + 1) >> 1); // ceil(i-i_low / 2)
            j_low = j;
            j = j + delta;
            i = i - delta;
        } else if (j > 0 && i < m && B[j-1] >= A[i]) {
            delta = ((j - j_low + 1) >> 1);
            i_low = i;
            i = i + delta;
            j = j - delta;
        } else
            break;
    }
    return i;
}


__device__
void RunMergeSequential(int* A, int m, int* B, int n, int* C) {
    int i = 0; // Index to A.
    int j = 0; // Index to B;
    int k = 0; // Index to C / output array;

    while ((i < m) && (j < n)) {
        if (A[i] <= B[j])
            C[k++] = A[i++];
        else
            C[k++] = B[j++];
    }
    if (i == m) {
        while (j < n)
            C[k++] = B[j++];
    } else {
        while (i < m)
            C[k++] = A[i++];
    }
}


__device__
int KernelCeil(int x, int y) {
    if (x % y == 0)
        return x / y;
    return x / y + 1;
}


__global__
void MergeBasicKernel(int* A, int m, int* B, int n, int* C) {
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    int elements_per_thread = KernelCeil(m + n, blockDim.x*gridDim.x);
    
    // Define the start and end output indices.
    int k_current = thread_index*elements_per_thread;
    int k_next = min(k_current+elements_per_thread, m+n);

    int i_current = ObtainCoRank(k_current, A, m, B, n);
    int i_next = ObtainCoRank(k_next, A, m, B, n);

    int j_current = k_current - i_current;
    int j_next = k_next - i_next;

    RunMergeSequential(
        &A[i_current],
        i_next - i_current,
        &B[j_current],
        j_next - j_current,
        &C[k_current]
    );
}


void runBasicMerge(int* A_h, int m, int* B_h, int n, int* C_h) {
    // Get size in bytes.
    size_t size_A = m * sizeof(int);
    size_t size_B = n * sizeof(int);
    size_t size_C = size_A + size_B;

    // Load and copy host variables to device memory.
    int *A_d, *B_d, *C_d;

    cudaMalloc((void***)&A_d, size_A);
    cudaMemcpy(A_d, A_h, size_A, cudaMemcpyHostToDevice);

    cudaMalloc((void***)&B_d, size_B);
    cudaMemcpy(B_d, B_h, size_B, cudaMemcpyHostToDevice);

    cudaMalloc((void***)&C_d, size_C);

    // Invoke kernel.
    dim3 dimBlock(THREADS_NUM_PER_BLOCK);
    dim3 dimGrid(1);
    MergeBasicKernel<<<dimGrid, dimBlock>>>(A_d, m, B_d, n, C_d);

    // Copy the output array from the device memory.
    cudaMemcpy(C_h, C_d, size_C, cudaMemcpyDeviceToHost);

    // Free device arrays.
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}


int main() {
    int m = 5, n = 4;
    int A[] = {1, 7, 8, 9, 10};
    int B[] = {7, 10, 10, 12};
    int C[m + n];

    runBasicMerge(A, m, B, n, C);

    // Print output.
    for (int i = 0; i < m+n; ++i) {
        printf("%d ", C[i]);
    }
    printf("\n");

    return 0;
}