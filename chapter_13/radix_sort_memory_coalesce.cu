/**
 *  Perform radix sort with memory coalescing, corresponds to chapter 13.4.
 *  Note that instead of using exclusive scan, the implementation here uses inclusive scan.
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define INPUT_LENGTH 16
#define THREADS_NUM_PER_BLOCK 4
#define BLOCK_NUM ceil(INPUT_LENGTH * 1.0 / THREADS_NUM_PER_BLOCK)


__device__
void runLocalInclusiveScan(
    unsigned int* array,
    unsigned int block_size
) {
    __shared__ unsigned int shared_array[THREADS_NUM_PER_BLOCK];

    if (threadIdx.x < block_size)
        shared_array[threadIdx.x] = array[threadIdx.x];
    else
        shared_array[threadIdx.x] = 0.0f;
    
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();

        unsigned int temp_value;
        if (threadIdx.x >= stride)
            temp_value = shared_array[threadIdx.x] + shared_array[threadIdx.x - stride];
        __syncthreads();

        if (threadIdx.x >= stride)
            shared_array[threadIdx.x] = temp_value;
    }

    if (threadIdx.x < block_size)
        array[threadIdx.x] = shared_array[threadIdx.x];
}


__device__
void runInclusiveScanAcrossBlocks(
    unsigned int* block_bucket_scan_values,
    unsigned int* block_scan_gate,
    unsigned int* block_flags
) {
    while (atomicAdd(&block_scan_gate[0], 0) < BLOCK_NUM - 1) {}
    if (blockIdx.x == 0) {
        block_bucket_scan_values[1] += block_bucket_scan_values[0];
        atomicAdd(&(block_flags[blockIdx.x]), 1);
    }
        
    unsigned int index1 = blockIdx.x * 2;
    unsigned int index2 = blockIdx.x * 2 + 1;
    if (blockIdx.x > 0) {
        // Wait for the previous flag; recall that atomicAdd returns the original value.
        while (atomicAdd(&block_flags[blockIdx.x - 1], 0) == 0) {}

        block_bucket_scan_values[index1] += block_bucket_scan_values[index1 - 1];;
        block_bucket_scan_values[index2] += block_bucket_scan_values[index1];
        __threadfence();

        // Update the flag.
        atomicAdd(&block_flags[blockIdx.x], 1);
    }
}


__global__
void RadixSortMemoryCoalescedKernel(
    unsigned int* input,
    unsigned int* output,
    unsigned int* block_bucket_scan_values,
    unsigned int* block_flags,
    unsigned int* block_scan_gate,
    unsigned int N,
    unsigned int iter
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    // To be used for storing the bits values and the inclusive scan results.
    __shared__ unsigned int local_bits_array[THREADS_NUM_PER_BLOCK];
    __shared__ unsigned int local_sorted_array[THREADS_NUM_PER_BLOCK];

    // Compute the last block "size" (i.e., valid last thread index + 1) given N.
    __shared__ unsigned int block_size;
    if (threadIdx.x == 0) {
        if ((blockIdx.x + 1)*blockDim.x > N)
            block_size = blockDim.x % N;
        else
            block_size = blockDim.x;
    }
    __syncthreads();

    // Store the bits value given the iteration.
    unsigned int key, bit;
    if (index < N) {
        key = input[index];
        bit = (key >> iter) & 1;
        local_bits_array[threadIdx.x] = bit;
    }
    __syncthreads();

    // Run local exclusive scan.
    runLocalInclusiveScan(local_bits_array, block_size);
    __syncthreads();

    // Sort locally.
    if (threadIdx.x < block_size) {
        unsigned int num_ones_before = (threadIdx.x == 0) ? 0 : local_bits_array[threadIdx.x - 1];
        unsigned int num_ones_total = local_bits_array[block_size - 1];
        unsigned int destination = (bit == 0) ? (threadIdx.x - num_ones_before)
                                              : (block_size - num_ones_total + num_ones_before);
        local_sorted_array[destination] = key;
    }

    // Compute the thread blocks' local bucket size.
    if (threadIdx.x == 0)
        block_bucket_scan_values[blockIdx.x] = block_size - local_bits_array[block_size-1];
        block_bucket_scan_values[blockIdx.x + gridDim.x] = local_bits_array[block_size-1];
        atomicAdd(&block_scan_gate[0], 1);
        runInclusiveScanAcrossBlocks(
            block_bucket_scan_values,
            block_scan_gate,
            block_flags
        );

    while (atomicAdd(&block_flags[gridDim.x - 1], 0) == 0) {}

    // Store the element in global memory.
    unsigned int beginning_position;
    if (threadIdx.x < block_size)
        bit = (local_sorted_array[threadIdx.x] >> iter) & 1;

    if (blockIdx.x == 0)
        beginning_position = (bit == 0) ? 0 : block_bucket_scan_values[gridDim.x - 1];
    else
        beginning_position = (bit == 0) ? block_bucket_scan_values[blockIdx.x - 1]
                                        : block_bucket_scan_values[blockIdx.x + gridDim.x - 1];
    
    if (threadIdx.x < block_size) {
        unsigned int global_destination = (bit == 0) ? threadIdx.x + beginning_position
                                                     : threadIdx.x - (block_size - local_bits_array[block_size - 1]) + beginning_position;

        output[global_destination] = local_sorted_array[threadIdx.x];
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
    unsigned int *input_d, *output_d, *block_bucket_scan_values_d, *block_flags_d, *block_scan_gate_d;

    cudaMalloc((void***)&input_d, size_array);
    cudaMemcpy(input_d, input_h, size_array, cudaMemcpyHostToDevice);

    cudaMalloc((void***)&output_d, size_array);
    cudaMalloc((void***)&block_bucket_scan_values_d, size_array_block*2);
    cudaMalloc((void***)&block_flags_d, size_array_block);
    cudaMalloc((void***)&block_scan_gate_d, sizeof(unsigned int));

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
        cudaMemset(block_bucket_scan_values_d, 0, size_array_block*2);
        cudaMemset(block_flags_d, 0, size_array_block);
        cudaMemset(block_scan_gate_d, 0, sizeof(unsigned int));
        
        RadixSortMemoryCoalescedKernel<<<dimGrid, dimBlock>>>(
            input_d, output_d,
            block_bucket_scan_values_d,
            block_flags_d,
            block_scan_gate_d,
            N, i);
        
        // Comment out the printing below for debugging purposes.
        // cudaMemcpy(output_h, output_d, size_array, cudaMemcpyDeviceToHost);
        // for (int j = 0; j < N; ++j) {
        //     printf("%d ", output_h[j]);
        // }
        // printf("\n");

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
    cudaFree(block_bucket_scan_values_d);
    cudaFree(block_flags_d);
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