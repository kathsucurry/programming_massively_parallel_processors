/**
 *  Perform radix sort, corresponds to chapter 13.3.
 *  Note that instead of using exclusive scan, the implementation here uses inclusive scan.
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define INPUT_LENGTH 16
#define SECTION_LENGTH 8
#define SUBSECTION_LENGTH 2
#define SCAN_THREADS_NUM_PER_BLOCK SECTION_LENGTH / SUBSECTION_LENGTH
#define SCAN_BLOCKS_PER_SECTION 1
#define SCAN_BLOCK_NUM INPUT_LENGTH / SECTION_LENGTH
#define SORT_THREADS_NUM_PER_BLOCK 4
#define SORT_BLOCK_NUM INPUT_LENGTH / SORT_THREADS_NUM_PER_BLOCK


__device__ unsigned int block_counter;
__device__ unsigned int block_scan_values[SCAN_BLOCK_NUM];
__device__ unsigned int flags[SCAN_BLOCK_NUM];


__global__
void InitializeGlobalDeviceVariables() {
    unsigned int index = threadIdx.x;
    if (index == 0)
        block_counter = 0;
    
    block_scan_values[index] = 0;
    flags[index] = 0;
}


__global__
void GenerateBitsFromArrayKernel(
    unsigned int* input,
    unsigned int* output,
    unsigned int N,
    unsigned int iter
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int key, bit;
    if (index < N) {
        key = input[index];
        bit = (key >> iter) & 1;
        output[index] = bit;
    }
}


/*
 * Perform single-pass exclusive scan using domino style.
 */
__global__
void SinglePassInclusiveScanKernel(
    unsigned int* input,
    unsigned int* output,
    unsigned int size
) {
    __shared__ unsigned int shared_array[SECTION_LENGTH];

    // Handle cases where blocks are not scheduled linearly according to blockIdx values.
    __shared__ unsigned int shared_block_id;
    if (threadIdx.x == 0)
        shared_block_id = atomicAdd(&block_counter, 1);
    __syncthreads();
    
    unsigned int block_id = shared_block_id;
    unsigned int section_index = shared_block_id / SCAN_BLOCKS_PER_SECTION;

    /* STAGE 1: PERFORM SCAN THROUGHOUT THE BLOCK/SECTION */

    // Phase 1.1: transfer data from global to shared memory.
    for (unsigned int iter = 0; iter < SUBSECTION_LENGTH; ++iter) {
        if (section_index*SECTION_LENGTH + iter*blockDim.x + threadIdx.x < size)
            shared_array[iter*blockDim.x + threadIdx.x] = input[section_index*SECTION_LENGTH + iter*blockDim.x + threadIdx.x];
        else
            shared_array[iter*blockDim.x + threadIdx.x] = 0.0f;
    }
    __syncthreads();

    // Phase 1.2: perform sequential scan in each subsection.
    for (unsigned int index = 1; index < SUBSECTION_LENGTH; ++index) {
        shared_array[threadIdx.x*SUBSECTION_LENGTH+index] += shared_array[threadIdx.x*SUBSECTION_LENGTH+index-1];

    }
    __syncthreads();

    // Phase 2: perform scan operation on the last element in each subsection.
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        float temp_value;
        if (threadIdx.x >= stride)
            temp_value = shared_array[(threadIdx.x+1)*SUBSECTION_LENGTH-1] + shared_array[(threadIdx.x-stride + 1)*SUBSECTION_LENGTH - 1];
        __syncthreads();

        if (threadIdx.x >= stride)
            shared_array[(threadIdx.x+1)*SUBSECTION_LENGTH - 1] = temp_value;
        __syncthreads();
    }

    // Phase 3: add the last element of its predecessor's subsection to its element except for the last element.
    for (unsigned int index = 0; index < SUBSECTION_LENGTH - 1; ++index) {
        if (threadIdx.x > 0)
            shared_array[threadIdx.x*SUBSECTION_LENGTH+index] += shared_array[threadIdx.x*SUBSECTION_LENGTH-1];
        __syncthreads();
    }

    if (threadIdx.x == blockDim.x - 1) {
        block_scan_values[block_id] = shared_array[(threadIdx.x+1) * SUBSECTION_LENGTH - 1];
        if (block_id == 0) {
            // Start the domino from block_id 0.
            atomicAdd(&(flags[block_id]), 1);
        }
    }

    // Ensure all block_scan_values have been updated.
    __syncthreads();

    /* STAGE 2: OBTAIN SUM VALUE FROM PREDECESSOR BLOCK */
    // Store the accumulated sum from the previous block.
    __shared__ float shared_previous_sum;
    if (threadIdx.x == 0 && block_id > 0) {
        // Wait for the previous flag; recall that atomicAdd returns the original value.
        while (atomicAdd(&flags[block_id - 1], 0) == 0) {}

        // Read the previous partial sum.
        shared_previous_sum = block_scan_values[block_id - 1];
        
        // Propagate the partial sum.
        block_scan_values[block_id] += shared_previous_sum;

        // Memory fence.
        __threadfence();

        // Update the flag.
        atomicAdd(&flags[block_id], 1);
    } else {
        shared_previous_sum = 0;
    }
    __syncthreads();

    for (unsigned int iter = 0; iter < SUBSECTION_LENGTH; ++iter) {
        if (section_index*SECTION_LENGTH + iter*blockDim.x + threadIdx.x < size)
            output[section_index*SECTION_LENGTH + iter*blockDim.x + threadIdx.x] = shared_array[iter*blockDim.x + threadIdx.x] + shared_previous_sum;
    }
}


__global__
void RadixSortKernel(
    unsigned int* input,
    unsigned int* output,
    unsigned int* scan_results,
    unsigned int N,
    unsigned int iter
){
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int key, bit;
    if (index < N) {
        key = input[index];
        bit = (key >> iter) & 1;
        unsigned int num_ones_before = (index == 0) ? 0 : scan_results[index - 1];
        unsigned int num_ones_total = scan_results[N - 1];
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

    // Load and copy host variables to device memory.
    unsigned int *input_d, *output_d, *bits_d, *scan_results_d;

    cudaMalloc((void***)&input_d, size_array);
    cudaMemcpy(input_d, input_h, size_array, cudaMemcpyHostToDevice);

    cudaMalloc((void***)&output_d, size_array);
    cudaMalloc((void***)&bits_d, size_array);
    cudaMalloc((void***)&scan_results_d, size_array);

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
    dim3 dimSortBlock(SORT_THREADS_NUM_PER_BLOCK);
    dim3 dimScanBlock(SCAN_THREADS_NUM_PER_BLOCK);
    dim3 dimSortSortGrid(SORT_BLOCK_NUM);
    dim3 dimScanGrid(ceil(N * 1.0 / SECTION_LENGTH * SCAN_BLOCKS_PER_SECTION));
    
    for (int i = 0; i < bit_counter; ++i) {
        InitializeGlobalDeviceVariables<<<1, SCAN_BLOCK_NUM>>>();
        GenerateBitsFromArrayKernel<<<dimSortSortGrid, dimSortBlock>>>(input_d, bits_d, N, i);
        SinglePassInclusiveScanKernel<<<dimScanGrid, dimScanBlock>>>(bits_d, scan_results_d, N);
        RadixSortKernel<<<dimSortSortGrid, dimSortBlock>>>(input_d, output_d, scan_results_d, N, i);
        cudaMemcpy(output_h, output_d, size_array, cudaMemcpyDeviceToHost);
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
    cudaFree(scan_results_d);
}


int main() {
    unsigned int input[] = {12, 3, 6, 9, 15, 8, 5, 10, 9, 6, 11, 13, 4, 10, 7, 0};
    // unsigned int input[] = {12, 6, 8, 10, 6, 4, 10, 0, 3, 9, 15, 5, 9, 11, 13, 7};
    // unsigned int input[] = {12, 8, 4, 0, 9, 5, 9, 13, 6, 10, 6, 10, 3, 15, 11, 7};
    // unsigned int input[] = {8, 0, 9, 9, 10, 10, 3, 11, 12, 4, 5, 13, 6, 6, 15, 7};
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