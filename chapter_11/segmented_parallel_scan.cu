/**
 *  Perform segmented parallel scan for arbitrary-length inputs, corresponds to chapter 11.6 and exercise 8.
 *  Utilizes Kogge-Stone algorithm for simplicity.
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define INPUT_LENGTH 4096
// Subsectioning is used for thread coarsening (chapter 11.5).
#define SECTION_LENGTH 32
#define SUBSECTION_LENGTH 4
#define BLOCKS_PER_SECTION 1
// Calculate the total number of threads (i.e., the number of subsections).
#define BLOCK_SIZE SECTION_LENGTH / SUBSECTION_LENGTH



/*
 * Kernel 1: the three-phase kernel that corresponds to chapter 11.5.
 * Performs on one section (consisting of multiple blocks with threads number = number of subsections).
 */
__global__
void ThreePhaseScanKernel(
    float* input_array,
    float* output_array,
    float* last_block_element_array,
    unsigned int size
) {
    __shared__ float shared_array[SECTION_LENGTH];
    unsigned int section_index = blockIdx.x / BLOCKS_PER_SECTION;

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

    for (unsigned int iter = 0; iter < SUBSECTION_LENGTH; ++iter) {
        if (section_index*SECTION_LENGTH + iter*blockDim.x + threadIdx.x < size)
            output_array[section_index*SECTION_LENGTH + iter*blockDim.x + threadIdx.x] = shared_array[iter*blockDim.x + threadIdx.x];
    }

    // Store the last element of each section.
    if (threadIdx.x == blockDim.x - 1)
        last_block_element_array[blockIdx.x] = shared_array[(threadIdx.x+1) * SUBSECTION_LENGTH - 1];
}



/*
 * Kernel 2: perform Kogge-Stone algorithm-based parallel scan on a single thread block.
 */
__global__
void KoggeStoneKernel(
    float* inplace_array,
    unsigned int size
) {
    __shared__ float shared_array[INPUT_LENGTH/SECTION_LENGTH];

    if (threadIdx.x < size)
        shared_array[threadIdx.x] = inplace_array[threadIdx.x];
    else
        shared_array[threadIdx.x] = 0.0f;

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();

        float temp_value;
        if (threadIdx.x >= stride)
            temp_value = shared_array[threadIdx.x] + shared_array[threadIdx.x - stride];
        __syncthreads();

        if (threadIdx.x >= stride)
            shared_array[threadIdx.x] = temp_value;
    }

    if (threadIdx.x < size)
        inplace_array[threadIdx.x] = shared_array[threadIdx.x];
}


/*
 * Kernel 3: update the output array using the scan block sum array.
 */
__global__
void UpdateOutputArrayKernel(
    float* output_array,
    float* last_block_sum_element_array,
    unsigned int size
) {
    if (blockIdx.x == 0)
        return;
    unsigned int block_offset = (blockIdx.x*blockDim.x + threadIdx.x)*SUBSECTION_LENGTH;
    for (unsigned int iter = 0; iter < SUBSECTION_LENGTH; ++iter) {
        if (block_offset + iter < size)
            output_array[block_offset + iter] += last_block_sum_element_array[blockIdx.x - 1];
    }
}



void runSegmentedParallelScan(
    float* input_array_h,
    float* output_array_h,
    unsigned int size
) {
    // Get size in bytes.
    size_t size_input = size * sizeof(float);
    size_t size_output = size * sizeof(float);

    // For allocating the value of last element in each block.
    size_t size_num_blocks = INPUT_LENGTH / SECTION_LENGTH * sizeof(float);

    // Load and copy host variables to device memory.
    float * input_array_d, * output_array_d, * last_block_element_array_d;
    
    cudaMalloc((void***)&input_array_d, size_input);
    cudaMemcpy(input_array_d, input_array_h, size_input, cudaMemcpyHostToDevice);

    cudaMalloc((void***)&output_array_d, size_output);

    cudaMalloc((void***)&last_block_element_array_d, size_num_blocks);

    // Invoke kernel 1.
    dim3 dimKernel1Block(BLOCK_SIZE);
    dim3 dimKernel1Grid(ceil(INPUT_LENGTH * 1.0 / SECTION_LENGTH * BLOCKS_PER_SECTION));
    ThreePhaseScanKernel<<<dimKernel1Grid, dimKernel1Block>>>(input_array_d, output_array_d, last_block_element_array_d, size);

    // Invoke kernel 2.
    dim3 dimKernel2Block(INPUT_LENGTH/SECTION_LENGTH);
    dim3 dimKernel2Grid(1);
    KoggeStoneKernel<<<dimKernel2Grid, dimKernel2Block>>>(last_block_element_array_d, INPUT_LENGTH/SECTION_LENGTH);

    // Invoke kernel 3.
    UpdateOutputArrayKernel<<<dimKernel1Grid, dimKernel1Block>>>(output_array_d, last_block_element_array_d, size);

    // Copy the output matrix from the device memory.
    cudaMemcpy(output_array_h, output_array_d, size_output, cudaMemcpyDeviceToHost);

    // Free device vectors.
    cudaFree(input_array_d);
    cudaFree(output_array_d);
    cudaFree(last_block_element_array_d);
}


int main() {
    unsigned int input_size = INPUT_LENGTH;
    float input_array[input_size];
    float output_array[input_size];

    for (int i = 1; i < input_size + 1; ++i) 
        input_array[i-1] = i % 31;

    runSegmentedParallelScan(
        input_array,
        output_array,
        input_size
    );

    // Print the last 3 values of the output array.
    printf("%.0f %.0f %.0f \n", output_array[input_size - 3], output_array[input_size - 2], output_array[input_size - 1]);

    return 0;
}