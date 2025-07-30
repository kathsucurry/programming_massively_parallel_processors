/**
 *  Perform radix sort allowing for multi-bits + thread coarsening, corresponds to chapter 13.6.
 */
#include <cmath>
#include <stdio.h>
#include <cuda_runtime.h>

#define INPUT_LENGTH 16
#define KEYS_NUM_PER_THREADS 2 // For thread coarsening.
#define THREADS_NUM_PER_BLOCK 4
#define BLOCK_NUM ceil(INPUT_LENGTH * 1.0 / THREADS_NUM_PER_BLOCK / KEYS_NUM_PER_THREADS)
#define NUM_BITS 2
#define BITS_POWER (int) std::pow(2, NUM_BITS)


__device__
void runLocalInclusiveScan(
    unsigned int* array,
    unsigned int block_size
) {
    __shared__ unsigned int shared_array[THREADS_NUM_PER_BLOCK*KEYS_NUM_PER_THREADS];
    __shared__ unsigned int shared_temp_array[THREADS_NUM_PER_BLOCK*KEYS_NUM_PER_THREADS];
    
    for (unsigned int key_iter = 0; key_iter < KEYS_NUM_PER_THREADS; ++key_iter) {
        unsigned int index = KEYS_NUM_PER_THREADS*threadIdx.x + key_iter;
        if (index < block_size)
            shared_array[index] = array[index];
        else
            shared_array[index] = 0.0f;
    }
    __syncthreads();

    for (unsigned int stride = 1; stride < block_size; stride *= 2) {
        for (unsigned int key_iter = 0; key_iter < KEYS_NUM_PER_THREADS; ++key_iter) {
            unsigned int index = KEYS_NUM_PER_THREADS*threadIdx.x + key_iter;
            unsigned int temp_value;
            if (index >= stride)
                temp_value = shared_array[index] + shared_array[index - stride];
                shared_temp_array[index] = temp_value;
        }
        __syncthreads();
        for (unsigned int key_iter = 0; key_iter < KEYS_NUM_PER_THREADS; ++key_iter) {
            unsigned int index = KEYS_NUM_PER_THREADS*threadIdx.x + key_iter;
            if (index >= stride)
                shared_array[index] = shared_temp_array[index];
        }
        __syncthreads();
    }

    for (unsigned int key_iter = 0; key_iter < KEYS_NUM_PER_THREADS; ++key_iter) {
        unsigned int index = KEYS_NUM_PER_THREADS*threadIdx.x + key_iter;
        if (index < block_size)
            array[index] = shared_array[index];
    }
}


__device__
void runInclusiveScanAcrossBlocks(
    unsigned int* block_scan_gate,
    unsigned int* block_bucket_scan_values,
    unsigned int* block_flags
) {
    // Ensure block_bucket_scan_values is ready to use.
    while (atomicAdd(&block_scan_gate[0], 0) < BLOCK_NUM - 1) {};
    
    if (blockIdx.x == 0) {
        for (int i = 1; i < BITS_POWER; ++i)
            block_bucket_scan_values[i] += block_bucket_scan_values[i - 1];
        __threadfence();
        atomicAdd(&(block_flags[blockIdx.x]), 1);
    }
        
    if (blockIdx.x > 0) {
        // Wait for the previous flag; recall that atomicAdd returns the original value.
        while (atomicAdd(&block_flags[blockIdx.x - 1], 0) == 0) {}

        for (int i = 0; i < BITS_POWER; ++i) {
            unsigned int index = blockIdx.x * BITS_POWER + i;
            block_bucket_scan_values[index] += block_bucket_scan_values[index - 1];
        }
        __threadfence();

        // Update the flag.
        atomicAdd(&block_flags[blockIdx.x], 1);
    }
    __syncthreads();
}


__global__
void RadixSortCoarsenedThreadsKernel(
    unsigned int* input,
    unsigned int* output,
    unsigned int* block_bucket_scan_values,
    unsigned int* block_flags,
    unsigned int* block_scan_gate,
    unsigned int N,
    unsigned int global_iter
) {
    // To be used for storing the bits values and the inclusive scan results.
    __shared__ unsigned int local_bits_array[THREADS_NUM_PER_BLOCK*KEYS_NUM_PER_THREADS];
    __shared__ unsigned int local_sorted_array[THREADS_NUM_PER_BLOCK*KEYS_NUM_PER_THREADS];
    __shared__ unsigned int local_temp_sorted_array[THREADS_NUM_PER_BLOCK*KEYS_NUM_PER_THREADS];
    extern __shared__ unsigned int local_bucket_size[];

    // Reset the values.
    if (threadIdx.x == 0)
        for (int i = 0; i < BITS_POWER; ++i)
            local_bucket_size[i] = 0;
    __syncthreads();
    
    // Compute the last block "size" (i.e., valid last thread index + 1) given N.
    __shared__ unsigned int block_size;
    if (threadIdx.x == 0) {
        if ((blockIdx.x + 1)*(blockDim.x*KEYS_NUM_PER_THREADS) > N)
            block_size = (blockDim.x*KEYS_NUM_PER_THREADS) % N;
        else
            block_size = (blockDim.x*KEYS_NUM_PER_THREADS);
    }
    __syncthreads();

    // Initialize local sorted array with the current input value; ensure memory coalescing.
    for (int i = 0; i < KEYS_NUM_PER_THREADS; ++i) {
        if (threadIdx.x + i * THREADS_NUM_PER_BLOCK < block_size) {
            unsigned int global_index = blockIdx.x * blockDim.x * KEYS_NUM_PER_THREADS + threadIdx.x;
            local_sorted_array[threadIdx.x + i * THREADS_NUM_PER_BLOCK] = input[global_index + i * THREADS_NUM_PER_BLOCK];
        }
    }
    __syncthreads();

    // Run local sort NUM_BITS time(s).
    for (int bit_iter = 0; bit_iter < NUM_BITS; ++bit_iter) {
        for (int key_iter = 0; key_iter < KEYS_NUM_PER_THREADS; ++key_iter) {
            unsigned int key_index = threadIdx.x * KEYS_NUM_PER_THREADS + key_iter;
            if (key_index < block_size) {
                local_bits_array[key_index] = (local_sorted_array[key_index] >> (NUM_BITS*global_iter) >> bit_iter) & 1;
            }
        }
        __syncthreads();
        
        // Run local exclusive scan.
        runLocalInclusiveScan(local_bits_array, block_size);
        __syncthreads();

        // Sort locally.
        for (int key_iter = 0; key_iter < KEYS_NUM_PER_THREADS; ++key_iter) {
            unsigned int key_index = threadIdx.x * KEYS_NUM_PER_THREADS + key_iter;
            unsigned int current_iter_bit = (local_sorted_array[key_index] >> (NUM_BITS*global_iter) >> bit_iter) & 1;
            unsigned int destination;
            if (key_index < block_size) {
                unsigned int num_ones_before = (key_index == 0) ? 0 : local_bits_array[key_index - 1];
                unsigned int num_ones_total = local_bits_array[block_size - 1];
                destination = (current_iter_bit == 0) ? (key_index - num_ones_before)
                                                      : (block_size - num_ones_total + num_ones_before);
                local_temp_sorted_array[destination] = local_sorted_array[key_index];
            }
        }
        __syncthreads();
        for (int key_iter = 0; key_iter < KEYS_NUM_PER_THREADS; ++key_iter) {
            unsigned int key_index = threadIdx.x * KEYS_NUM_PER_THREADS + key_iter;
            if (key_index < block_size)
                local_sorted_array[key_index] = local_temp_sorted_array[key_index];
        }
        __syncthreads(); 
    }

    // Store the thread blocks' local bucket size.
    for (int key_iter = 0; key_iter < KEYS_NUM_PER_THREADS; ++key_iter) {
        unsigned int key_index = threadIdx.x * KEYS_NUM_PER_THREADS + key_iter;
        unsigned int bit = (local_sorted_array[key_index] >> (NUM_BITS*global_iter)) & (BITS_POWER - 1);
        atomicAdd(&local_bucket_size[bit], 1);
        atomicAdd(&block_bucket_scan_values[gridDim.x*bit + blockIdx.x], 1);
    }
    __syncthreads();

    // Perform a sequantial scan on local_bucket_size.
    if (threadIdx.x == 0)
        for (int i = 1; i < BITS_POWER; ++i)
            local_bucket_size[i] += local_bucket_size[i-1];

    if (threadIdx.x == 0) {
        atomicAdd(&block_scan_gate[0], 1);
        runInclusiveScanAcrossBlocks(
            block_scan_gate,
            block_bucket_scan_values,
            block_flags
        );
    }
    while (atomicAdd(&block_flags[gridDim.x - 1], 0) == 0) {}

    // Store the element in global memory; ensure memory coalescing.
    for (int key_iter = 0; key_iter < KEYS_NUM_PER_THREADS; ++key_iter) {
        unsigned int key_index = threadIdx.x + key_iter * THREADS_NUM_PER_BLOCK;
        unsigned int beginning_position;
        unsigned int bit = local_sorted_array[key_index] >> (NUM_BITS*global_iter) & (BITS_POWER - 1);
        if (blockIdx.x == 0)
            beginning_position = (bit == 0) ? 0 : block_bucket_scan_values[bit*gridDim.x - 1];
        else
            beginning_position = block_bucket_scan_values[blockIdx.x + bit*gridDim.x - 1];

        if (key_index < block_size) {
            unsigned int destination = (bit == 0) ? key_index + beginning_position
                                                  : key_index - local_bucket_size[bit-1] + beginning_position;

            output[destination] = local_sorted_array[key_index];
        }
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
    size_t size_bucket = BITS_POWER * BLOCK_NUM * sizeof(unsigned int);

    // Load and copy host variables to device memory.
    unsigned int *input_d, *output_d, *block_bucket_scan_values_d, *block_flags_d, *block_scan_gate_d;

    cudaMalloc((void***)&input_d, size_array);
    cudaMemcpy(input_d, input_h, size_array, cudaMemcpyHostToDevice);

    cudaMalloc((void***)&output_d, size_array);
    cudaMalloc((void***)&block_bucket_scan_values_d, size_bucket);
    cudaMalloc((void***)&block_flags_d, size_array_block);
    cudaMalloc((void***)&block_scan_gate_d , sizeof(unsigned int));

    // Get the total number of iterations.
    // 1. Get the max value across all input elements.
    unsigned int max_value = 0;
    for (int i = 0; i < N; ++i)
        max_value = max(max_value, input_h[i]);
    // 2. Get the number of bits.
    unsigned int bit_counter = 0;
    while (max_value > 0) {
        max_value = max_value >> NUM_BITS;
        ++bit_counter;
    }

    // Invoke kernel per iteration.
    dim3 dimBlock(THREADS_NUM_PER_BLOCK);
    dim3 dimGrid(BLOCK_NUM);
    
    for (int i = 0; i < bit_counter; ++i) {
        // Initialize the block scan values and flags for each iteration.
        cudaMemset(block_bucket_scan_values_d, 0, size_bucket);
        cudaMemset(block_flags_d, 0, size_array_block);
        cudaMemset(block_scan_gate_d, 0, sizeof(unsigned int));
        
        RadixSortCoarsenedThreadsKernel<<<dimGrid, dimBlock, BITS_POWER * sizeof(unsigned int)>>>(
            input_d, output_d,
            block_bucket_scan_values_d,
            block_flags_d,
            block_scan_gate_d,
            N,
            i
        );
        
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
    cudaFree(block_scan_gate_d);
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