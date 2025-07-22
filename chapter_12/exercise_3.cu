/**
 * Perform tiled merge kernel with modifications, corresponds to exercise 3.
 * The modifications include:
 * 1. MODIFICATION 1: Reduce the shared memory length to just tile_size instead of 2*tile_size.
 * 2. MODIFICATION 2: Call co-rank function to get the A and B elements to be consumed.
 * 
 */

#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

#define THREADS_NUM_PER_BLOCK 3
#define BLOCKS_NUM_PER_GRID 4
#define TILE_SIZE 4


__device__
int runKernelDivCeil(int x, int y) {
    if (x % y == 0)
        return x / y;
    return x / y + 1;
}


/**
 * Retrieve the co-rank using binary search, i.e., the begining positions of A and B that will be
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
int obtainCoRank(int k, int* A, int m, int* B, int n) {
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
            delta = ((i - i_low + 1) >> 1);
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
void runMergeSequential(int* A, int m, int* B, int n, int* C) {
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


__global__
void MergeTiledKernel(int* A, int m, int* B, int n, int* C, int tile_size) {
    // Allocate shared memory.
    // MODIFICATION 1.2: at this point, we don't know where the B_shared should start.
    extern __shared__ int AB_shared[];
    int *A_shared = &AB_shared[0];
    int *B_shared;

    // Define the start and end point of the block's C subarray.
    int C_current = blockIdx.x * runKernelDivCeil(m+n, gridDim.x);
    int C_next = min((blockIdx.x+1) * runKernelDivCeil(m+n, gridDim.x), m+n);

    if (threadIdx.x == 0) {
        // Obtain the block-level co-rank values and make them visible to other threads.
        // Note that we only need these values at the beginning until they're used by
        // A_current and A_next.
        AB_shared[0] = obtainCoRank(C_current, A, m, B, n);
        AB_shared[1] = obtainCoRank(C_next, A, m, B, n);
    }
    // Ensure AB_shared updates are visible to all threads.
    __syncthreads();

    int A_current = AB_shared[0]; // start index of A in the block.
    int A_next = AB_shared[1];

    int B_current = C_current - A_current; // start index of B in the block.
    int B_next = C_next - A_next;
    __syncthreads();

    int iter_counter = 0;
    int C_length = C_next - C_current;
    int A_length = A_next - A_current;
    int B_length = B_next - B_current;
    int iter_total = runKernelDivCeil(C_length, tile_size);

    int C_completed = 0;
    // Initialize he total number A or B elements that have been consumed by all threads
    // of the block during the previous iterations of the while loop.
    int A_consumed = 0; 
    int B_consumed = 0;

    while (iter_counter < iter_total) {
        // To be used by the last iteration where the remaining elements < tile_size.
        int updated_tile_size = min(tile_size, C_length - C_completed);

        // MODIFICATION 2.
        int A_to_be_consumed = obtainCoRank(
            updated_tile_size,
            &A[A_current+A_consumed], min(updated_tile_size, A_length-A_consumed),
            &B[B_current+B_consumed], min(updated_tile_size, B_length-B_consumed));
        int B_to_be_consumed = updated_tile_size - A_to_be_consumed;

        B_shared = &AB_shared[A_to_be_consumed];

        for (int i = 0; i < A_to_be_consumed; i += blockDim.x) {
            if (i + threadIdx.x < A_length - A_consumed)
                A_shared[i + threadIdx.x] = A[A_current + A_consumed + i + threadIdx.x];
        }
        for (int i = 0; i < B_to_be_consumed; i += blockDim.x) {
            if (i + threadIdx.x < B_length - B_consumed)
                B_shared[i + threadIdx.x] = B[B_current + B_consumed + i + threadIdx.x];
        }
        __syncthreads();

        // Compute c_current and c_next for the individual thread.
        // tile_size / blockDim.x produces the number of elements per thread.
        int c_current = threadIdx.x  * (updated_tile_size/blockDim.x);
        int c_next = (threadIdx.x+1) * (updated_tile_size/blockDim.x);

        // Handle cases where updated_tile_size is not divisible by the number of threads.
        if (updated_tile_size % blockDim.x > 0) {
            if (threadIdx.x > 0)
                c_current += updated_tile_size % blockDim.x;
            c_next += updated_tile_size % blockDim.x;
        }

        c_current = (c_current <= C_length - C_completed) ? c_current : C_length - C_completed;
        c_next = (c_next <= C_length - C_completed) ? c_next : C_length - C_completed;

        // Find co-rank for c_current and c_next.
        int a_current = obtainCoRank(c_current, A_shared, A_to_be_consumed,
                                        B_shared, B_to_be_consumed);
        int b_current = c_current - a_current;

        int a_next = obtainCoRank(c_next, A_shared, A_to_be_consumed,
                                        B_shared, B_to_be_consumed);
        int b_next = c_next - a_next;
        
        // All threads call the sequential merge function.
        runMergeSequential(&A_shared[a_current], a_next-a_current, &B_shared[b_current], b_next-b_current,
                                &C[C_current+C_completed+c_current]);
        
        // Update the number of A and B elements that have been consumed so far.
        iter_counter++;
        C_completed += updated_tile_size;
        A_consumed += A_to_be_consumed;
        B_consumed = C_completed - A_consumed;
        __syncthreads();
    }
}


void runMerge(int* A_h, int m, int* B_h, int n, int* C_h) {
    // Get size in bytes.
    size_t size_A = m * sizeof(int);
    size_t size_B = n * sizeof(int);
    size_t size_C = size_A + size_B;
    // MODIFICATION 1.1.
    unsigned shared_memory_size = sizeof(int) * TILE_SIZE;

    // Load and copy host variables to device memory.
    int *A_d, *B_d, *C_d;

    cudaMalloc((void***)&A_d, size_A);
    cudaMemcpy(A_d, A_h, size_A, cudaMemcpyHostToDevice);

    cudaMalloc((void***)&B_d, size_B);
    cudaMemcpy(B_d, B_h, size_B, cudaMemcpyHostToDevice);

    cudaMalloc((void***)&C_d, size_C);

    // Invoke kernel.
    dim3 dimBlock(THREADS_NUM_PER_BLOCK);
    dim3 dimGrid(BLOCKS_NUM_PER_GRID);
    MergeTiledKernel<<<dimGrid, dimBlock, shared_memory_size>>>(A_d, m, B_d, n, C_d, TILE_SIZE);

    // Copy the output array from the device memory.
    cudaMemcpy(C_h, C_d, size_C, cudaMemcpyDeviceToHost);

    // Free device arrays.
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}


int compareIntPointer(const void *a, const void *b) {
    return ( *(int*)a - *(int*)b );
}


int main() {
    int m = 31, n = 41;
    int A[] = {2, 2, 2, 5, 5, 7, 7, 10, 13, 14, 17, 19, 21, 24, 27, 28, 29, 32, 33, 35, 38, 41, 41, 41, 42, 45, 46, 47, 49, 52, 83};
    int B[] = {1, 5, 7, 9, 10, 12, 13, 14, 17, 20, 20, 23, 26, 29, 32, 33, 36, 36, 36, 39, 41, 41, 44, 44, 44, 44, 44, 46, 47, 49, 52, 53, 55, 57, 57, 60, 62, 65, 67, 67, 74};
    int C[m + n];
    int C_expected[m + n];

    // Generate the expected C.
    for (int i = 0; i < m+n; ++i) {
        if (i >= m) {
            C_expected[i] = B[i - m];
        } else {
            C_expected[i] = A[i];
        }
    }
    std::qsort(C_expected, m+n, sizeof(int), compareIntPointer);

    runMerge(A, m, B, n, C);

    // Compare output.
    bool all_identical = true;
    for (int i = 0; i < m+n; ++i) {
        if (C[i] != C_expected[i]) {
            printf("The element with index %d is different: %d %d!", i, C[i], C_expected[i]);
            all_identical = false;
            break;
        }
    }
    if (all_identical)
        printf("All elements are identical; the kernel implementation should be correct.");

    printf("\n");

    // Comment out for more thorough check if m+n is small enough.
    // for (int i = 0; i < m+n; ++i) {
    //     printf("%d %d %d\n", C[i], C_expected[i], C[i] == C_expected[i]);
    // }
    return 0;
}