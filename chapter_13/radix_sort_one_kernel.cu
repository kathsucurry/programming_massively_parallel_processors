/**
 *  Perform radix sort, corresponds to chapter 13.3.
 *  Note that instead of using exclusive scan, the implementation here uses inclusive scan.
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define INPUT_LENGTH 16
#define THREADS_NUM_PER_BLOCK 4
#define BLOCK_NUM ceil(INPUT_LENGTH * 1.0 / THREADS_NUM_PER_BLOCK)


/*
 * Perform single-pass exclusive scan using domino style without thread coarsening.
 */
__device__
void runSinglePassInclusiveScan(
    unsigned int* bits,
    unsigned int* block_scan_values,
    unsigned int* block_flags,
    unsigned int N
) {
    __shared__ unsigned int shared_array[THREADS_NUM_PER_BLOCK];
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

    unsigned int shared_block_id = blockIdx.x;

    /* STAGE 1: PERFORM SCAN THROUGHOUT THE BLOCK/SECTION */
    // Phase 1: transfer data from global to shared memory.
    if (index < N)
        shared_array[threadIdx.x] = bits[index];
    else
        shared_array[threadIdx.x] = 0.0f;
    __syncthreads();

    // Phase 2: perform scan operation on the elements.
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        float temp_value;
        if (threadIdx.x >= stride)
            temp_value = shared_array[threadIdx.x] + shared_array[threadIdx.x-stride];
        __syncthreads();

        if (threadIdx.x >= stride)
            shared_array[threadIdx.x] = temp_value;
        __syncthreads();
    }

    // Store the last scan value per block.
    if (threadIdx.x == blockDim.x - 1) {
        block_scan_values[shared_block_id] = shared_array[threadIdx.x];
        if (shared_block_id == 0) {
            // Start the domino from block_id 0.
            atomicAdd(&(block_flags[shared_block_id]), 1);
        }
    }

    // Ensure all block_scan_values have been updated.
    __syncthreads();

    /* STAGE 2: OBTAIN SUM VALUE FROM PREDECESSOR BLOCK */
    // Store the accumulated sum from the previous block.
    __shared__ float shared_previous_sum;
    if (threadIdx.x == 0 && shared_block_id > 0) {
        // Wait for the previous flag; recall that atomicAdd returns the original value.
        while (atomicAdd(&block_flags[shared_block_id - 1], 0) == 0) {}

        // Read the previous partial sum.
        shared_previous_sum = block_scan_values[shared_block_id - 1];
        
        // Propagate the partial sum.
        block_scan_values[shared_block_id] += shared_previous_sum;

        // Memory fence.
        __threadfence();

        // Update the flag.
        atomicAdd(&block_flags[shared_block_id], 1);
    } else {
        shared_previous_sum = 0;
    }
    __syncthreads();

    if (threadIdx.x < N)
        bits[index] = shared_array[threadIdx.x] + shared_previous_sum;
}


__global__
void RadixSortKernel(
    unsigned int* input,
    unsigned int* output,
    unsigned int* bits,
    unsigned int* block_scan_values,
    unsigned int* block_flags,
    unsigned int* block_gate,
    unsigned int N,
    unsigned int iter
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int key, bit;
    if (index < N) {
        key = input[index];
        bit = (key >> iter) & 1;
        bits[index] = bit;
    }
    __syncthreads();

    runSinglePassInclusiveScan(bits, block_scan_values, block_flags, N);
    __threadfence();

    if (threadIdx.x == blockDim.x - 1) {
        atomicAdd(&block_gate[0], 1);
    }
    
    while (atomicAdd(&block_gate[0], 0) < gridDim.x) {}

    if (index < N) {
        unsigned int num_ones_before = (index == 0) ? 0 : bits[index - 1];
        unsigned int num_ones_total = bits[N - 1];
        unsigned int destination = (bit == 0) ? (index - num_ones_before)
                                              : (N - num_ones_total + num_ones_before);
        output[destination] = key;
    }
}


void runRadixSort(
    unsigned int* input_h,
    unsigned int* output_h,
    unsigned int N
) {
    // Get size in bytes.
    size_t size_array = N * sizeof(unsigned int);
    size_t size_array_block = BLOCK_NUM * sizeof(unsigned int);

    // Load and copy host variables to device memory.
    unsigned int *input_d, *output_d, *bits_d, *block_scan_values_d, *block_flags_d, *block_gate_d;

    cudaMalloc((void***)&input_d, size_array);
    cudaMemcpy(input_d, input_h, size_array, cudaMemcpyHostToDevice);

    cudaMalloc((void***)&output_d, size_array);
    cudaMalloc((void***)&bits_d, size_array);
    cudaMalloc((void***)&block_scan_values_d, size_array_block);
    cudaMalloc((void***)&block_flags_d, size_array_block);
    cudaMalloc((void***)&block_gate_d, sizeof(unsigned int));

    // Get the total number of iterations.
    // 1. Get the max value across all input elements.
    unsigned int max_value = 0;
    for (int i = 0; i < N; ++i)
        max_value = max(max_value, input_h[i]);
    // 2. Get the number of bits.
    unsigned int bit_counter = 0;
    while (max_value > 0) {
        max_value = max_value >> 1;
        ++bit_counter;
    }

    // Invoke kernel per iteration.
    dim3 dimBlock(THREADS_NUM_PER_BLOCK);
    dim3 dimGrid(BLOCK_NUM);
    
    for (int i = 0; i < bit_counter; ++i) {
        // Initialize the block scan values and flags for each iteration.
        cudaMemset(block_scan_values_d, 0, size_array_block);
        cudaMemset(block_flags_d, 0, size_array_block);
        cudaMemset(block_gate_d, 0, sizeof(unsigned int));
        
        RadixSortKernel<<<dimGrid, dimBlock>>>(input_d, output_d, bits_d, block_scan_values_d, block_flags_d, block_gate_d, N, i);
        
        // Comment out the printing below for debugging purposes.
        // cudaMemcpy(output_h, output_d, size_array, cudaMemcpyDeviceToHost);
        //     for (int j = 0; j < N; ++j) {
        //         printf("%d ", output_h[j]);
        //     }
        //     printf("\n");
        
        if (i < bit_counter - 1) {
            // Swap input_d and output_d memory location such that input_d stores the values (and memory address) of output_d.
            unsigned int* temp = input_d;
            input_d = output_d;
            output_d = temp;
        } 

    }

    // Copy the output array from the device memory.
    cudaMemcpy(output_h, output_d, size_array, cudaMemcpyDeviceToHost);

    // Free device arrays.
    cudaFree(input_d);
    cudaFree(output_d);
    cudaFree(bits_d);
    cudaFree(block_scan_values_d);
    cudaFree(block_flags_d);
    cudaFree(block_gate_d);
}


int main() {
    unsigned int input[] = {12, 3, 6, 9, 15, 8, 5, 10, 9, 6, 11, 13, 4, 10, 7, 0};
    unsigned int N = INPUT_LENGTH;
    unsigned int output[N];

    runRadixSort(input, output, N);

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

    return 0;
}