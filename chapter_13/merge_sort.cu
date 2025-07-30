/**
 *  Perform merge sort, corresponds to chapter 13.7.
 */
#include <cmath>
#include <stdio.h>
#include <cuda_runtime.h>

#define INPUT_LENGTH 16
#define THREADS_NUM_PER_BLOCK 4
#define BLOCK_NUM ceil(INPUT_LENGTH * 1.0 / THREADS_NUM_PER_BLOCK)


__device__
int runKernelDivCeil(int x, int y) {
    if (x % y == 0)
        return x / y;
    return x / y + 1;
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


__device__
void runSimpleMergeSort(int* input, unsigned int N) {
    int* output = (int*)malloc(N * sizeof(int));
    for (unsigned int stride = 2; stride <= N; stride *= 2) {
        unsigned int start_index = 0;
        while (start_index < N) {
            unsigned int size1 = stride / 2;
            unsigned int size2 = min(stride / 2, N - start_index);
            runMergeSequential(&input[start_index], size1, &input[start_index+size1], size2, &output[start_index]);
            start_index += stride;
        }
        for (int i = 0; i < N; ++i)
            input[i] = output[i];
    }
    free(output);
}


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
void runMerge(
    int* A, int m,
    int* B, int n,
    int* C, int C_block_start,
    int block_size
) { 
    __shared__ int A_shared[THREADS_NUM_PER_BLOCK];
    __shared__ int B_shared[THREADS_NUM_PER_BLOCK];

    // Retrieve the range of A and B of the block.
    __shared__ int i_block_current, i_block_next;
    __shared__ int j_block_current, j_block_next;
    if (threadIdx.x == 0) {
        i_block_current = obtainCoRank(C_block_start, A, m, B, n);
        i_block_next = obtainCoRank(C_block_start + block_size, A, m, B, n);

        j_block_current = C_block_start - i_block_current;
        j_block_next = C_block_start + block_size - i_block_next;
    }
    __syncthreads();

    if (threadIdx.x < i_block_next - i_block_current)
        A_shared[threadIdx.x] = A[i_block_current + threadIdx.x];
    
    if (threadIdx.x < j_block_next - j_block_current)
        B_shared[threadIdx.x] = B[j_block_current + threadIdx.x];
    __syncthreads();

    // Define the start and end output indices.
    int k_current = threadIdx.x;
    int k_next = threadIdx.x + 1;

    int i_current = obtainCoRank(k_current, A_shared, i_block_next - i_block_current, B_shared, j_block_next - j_block_current);
    int i_next = obtainCoRank(k_next, A_shared, i_block_next - i_block_current, B_shared, j_block_next - j_block_current);

    int j_current = k_current - i_current;
    int j_next = k_next - i_next;

    runMergeSequential(
        &A_shared[i_current],
        i_next - i_current,
        &B_shared[j_current],
        j_next - j_current,
        &C[C_block_start + k_current]
    );
}


__global__
void MergeSortKernel(
    int* input,
    int* output,
    unsigned int N,
    int* block_gates
) {
    __shared__ int shared_array[THREADS_NUM_PER_BLOCK];
    __shared__ unsigned int block_size;
    int *block_gate_prepare_input = &block_gates[0];
    int *block_gate_finish_merge = &block_gates[1];
    int *block_gate_update_input = &block_gates[2];


    unsigned int global_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x == 0) {
        if ((blockIdx.x + 1)*(blockDim.x) > N)
            block_size = blockDim.x % N;
        else
            block_size = blockDim.x;
    }
    __syncthreads();

    // Store the input values in the shared array.
    if (threadIdx.x < block_size)
        shared_array[threadIdx.x] = input[global_index];
    __syncthreads();

    // Sort the elements in shared array.
    if (threadIdx.x == 0)
        runSimpleMergeSort(shared_array, block_size);
    __syncthreads();

    // Update a global memory such that other blocks can access them.
    if (threadIdx.x < block_size)
        input[global_index] = shared_array[threadIdx.x];
    __syncthreads();
    
    if (threadIdx.x == 0) {
        __threadfence();
        atomicAdd(block_gate_prepare_input, 1);
    }
    while (atomicAdd(block_gate_prepare_input, 0) < gridDim.x) {}

    // Determine the size of the merge output C.
    unsigned int current_merge_size = 2 * blockDim.x;
    unsigned int counter = 0;
    while (current_merge_size <= N) {
        // Compute the start of the C range the block would output.
        unsigned int start_block_index = blockDim.x * blockIdx.x;
        // Given the block index, which C is it involved in?
        unsigned int C_index = start_block_index / current_merge_size;
        unsigned int start_index1 = C_index * current_merge_size;
        unsigned int size1 = current_merge_size / 2;
        unsigned int start_index2 = start_index1 + size1;
        unsigned int size2 = min(current_merge_size / 2, N - start_index2);
        runMerge(
            &input[start_index1],
            size1,
            &input[start_index2],
            size2,
            &output[start_index1],
            start_block_index - start_index1,
            block_size
        );
        __syncthreads();

        // Make sure all blocks are finished.
        if (threadIdx.x == 0) {
            __threadfence();
            atomicAdd(block_gate_finish_merge, 1);
        }
        while (atomicAdd(block_gate_finish_merge, 0) < (counter + 1) * gridDim.x) {}

        // Store the output elements in input.
        if (current_merge_size * 2 <= N) {
            if (threadIdx.x < block_size)
                input[global_index] = output[global_index];
            __syncthreads();

            if (threadIdx.x == 0) {
                __threadfence();
                atomicAdd(block_gate_update_input, 1);
            }
            while (atomicAdd(block_gate_update_input, 0) < (counter + 1) * gridDim.x) {}
        }
            
        ++counter;
        current_merge_size = current_merge_size * 2;
    }

}


void runMergeSort(
    int* input_h,
    int* output_h,
    unsigned int N
) {
    // Get size in bytes.
    size_t size_array = N * sizeof(int);

    // Load and copy host variables to device memory.
    int *input_d, *output_d, *block_gate_d;

    cudaMalloc((void***)&input_d, size_array);
    cudaMemcpy(input_d, input_h, size_array, cudaMemcpyHostToDevice);

    cudaMalloc((void***)&output_d, size_array);
    cudaMalloc((void***)&block_gate_d, 3 * sizeof(unsigned int));

    // Invoke kernel per iteration.
    dim3 dimBlock(THREADS_NUM_PER_BLOCK);
    dim3 dimGrid(BLOCK_NUM);
    
    MergeSortKernel<<<dimGrid, dimBlock>>>(input_d, output_d, N, block_gate_d);

    // Copy the output array from the device memory.
    cudaMemcpy(output_h, output_d, size_array, cudaMemcpyDeviceToHost);

    // Free device arrays.
    cudaFree(input_d);
    cudaFree(output_d);
}


int main() {
    int input[] = {12, 3, 6, 9, 15, 8, 5, 10, 9, 6, 11, 13, 4, 10, 7, 0};
    unsigned int N = INPUT_LENGTH;
    int output[N];

    runMergeSort(input, output, N);

    // Check if the output is properly sorted.
    bool is_properly_sorted = true;
    for (int i = 1; i < N; ++i) {
        if (output[i] < output[i-1]) {
            printf("The output is not sorted!\n");
            is_properly_sorted = false;
            break;
        }
    }
    if (is_properly_sorted)
        printf("The output is sorted :) \n");

    // Comment out to print all sorted elements.
    for (int i = 0; i < N; ++i)
        printf("%d ", output[i]);
    printf("\n");

    return 0;
}