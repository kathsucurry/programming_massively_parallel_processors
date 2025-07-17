#include <stdio.h>
#include <cuda_runtime.h>

#define INPUT_LENGTH 4096
// Subsectioning is used for thread coarsening (chapter 11.5).
#define SECTION_LENGTH 32
#define SUBSECTION_LENGTH 4
#define BLOCKS_PER_SECTION 1
// Calculate the total number of threads (i.e., the number of subsections).
#define BLOCK_SIZE SECTION_LENGTH / SUBSECTION_LENGTH
#define BLOCK_NUMBER INPUT_LENGTH / SECTION_LENGTH

__device__ unsigned int block_counter;
__device__ float block_scan_values[BLOCK_NUMBER];
__device__ unsigned int flags[BLOCK_NUMBER];


/*
 * Perform single-pass scan using domino style, corresponds to chapter 11.7.
 */
__global__
void SinglePassScanKernel(
    float* input_array,
    float* output_array,
    unsigned int size
) {
    __shared__ float shared_array[SECTION_LENGTH];

    // Handle cases where blocks are not scheduled linearly according to blockIdx values.
    __shared__ unsigned int shared_block_id;
    if (threadIdx.x == 0)
        shared_block_id = atomicAdd(&block_counter, 1);
    __syncthreads();
    unsigned int block_id = shared_block_id;

    unsigned int section_index = shared_block_id / BLOCKS_PER_SECTION;

    /* STAGE 1: PERFORM SCAN THROUGHOUT THE BLOCK/SECTION */

    // Phase 1.1: transfer data from global to shared memory.
    for (unsigned int iter = 0; iter < SUBSECTION_LENGTH; ++iter) {
        if (section_index*SECTION_LENGTH + iter*blockDim.x + threadIdx.x < size)
            shared_array[iter*blockDim.x + threadIdx.x] = input_array[section_index*SECTION_LENGTH + iter*blockDim.x + threadIdx.x];
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
            output_array[section_index*SECTION_LENGTH + iter*blockDim.x + threadIdx.x] = shared_array[iter*blockDim.x + threadIdx.x] + shared_previous_sum;
    }
}


void runSinglePassScan(
    float* input_array_h,
    float* output_array_h,
    unsigned int size
) {
    // Get size in bytes.
    size_t size_input = size * sizeof(float);
    size_t size_output = size * sizeof(float);

    // Load and copy host variables to device memory.
    float * input_array_d, * output_array_d;
    
    cudaMalloc((void***)&input_array_d, size_input);
    cudaMemcpy(input_array_d, input_array_h, size_input, cudaMemcpyHostToDevice);

    cudaMalloc((void***)&output_array_d, size_output);

    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGris(ceil(INPUT_LENGTH * 1.0 / SECTION_LENGTH * BLOCKS_PER_SECTION));
    SinglePassScanKernel<<<dimGris, dimBlock>>>(
        input_array_d,
        output_array_d,
        size
    );

    // Copy the output matrix from the device memory.
    cudaMemcpy(output_array_h, output_array_d, size_output, cudaMemcpyDeviceToHost);

    // Free device vectors.
    cudaFree(input_array_d);
    cudaFree(output_array_d);
}


int main() {
    unsigned int input_size = INPUT_LENGTH;
    float input_array[input_size];
    float output_array[input_size];

    for (int i = 1; i < input_size + 1; ++i) 
        input_array[i-1] = i % 31;

    runSinglePassScan(
        input_array,
        output_array,
        input_size
    );

    // Print the last 3 values of the output array.
    printf("%.0f %.0f %.0f \n", output_array[input_size - 3], output_array[input_size - 2], output_array[input_size - 1]);

    return 0;
}