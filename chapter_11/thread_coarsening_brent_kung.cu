#include <stdio.h>
#include <cuda_runtime.h>

#define SECTION_SIZE 2048
#define SUBSECTION_SIZE 4
#define BLOCK_SIZE SECTION_SIZE / SUBSECTION_SIZE / 2
// Here we assume that input size is the same as section size for simplicity.
#define INPUT_SIZE SECTION_SIZE


/**
 *  Perform Brent-Kung kernel for inclusive (segmented) scan with thread coarsening, corresponds to chapter 11.5.
 *  Assume only 1 block is needed in total.
 */
__global__
void BrentKungScanWithThreadCoarseningKernel(
    float* input_array,
    float* output_array,
    unsigned int size
) {
    __shared__ float shared_array[SECTION_SIZE];
    unsigned int global_start_index = threadIdx.x*SUBSECTION_SIZE;

    // Phase 1.1: transfer data from global to shared memory.
    // Recall that two elements are loaded at the beginning in Brent-Kung algorithm.
    for (unsigned int iter = 0; iter < SUBSECTION_SIZE*2; ++iter) {
        unsigned int index = iter*blockDim.x + threadIdx.x;
        if (index < size)
            shared_array[index] = input_array[index];
        else
            shared_array[index] = 0.0f;
    }
    __syncthreads();

    // Phase 1.2: perform sequential scan in each subsection.
    for (unsigned int index = 1; index < SUBSECTION_SIZE; ++index) {
        if (global_start_index+index < size)
            shared_array[global_start_index+index] += shared_array[global_start_index+index-1];
        if (global_start_index+index + blockDim.x*SUBSECTION_SIZE < size)
            shared_array[global_start_index+index + blockDim.x*SUBSECTION_SIZE] += shared_array[global_start_index+index + blockDim.x*SUBSECTION_SIZE-1];


    }
    __syncthreads();

    // Phase 2: perform scan operation on the last element in each subsection.
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        __syncthreads();
        unsigned int subsection_index = (threadIdx.x + 1)*2*stride - 1;
        if (subsection_index < blockDim.x*2)
            shared_array[(subsection_index+1)*SUBSECTION_SIZE-1] += shared_array[(subsection_index-stride+1)*SUBSECTION_SIZE-1];
    }

    for (int stride = SECTION_SIZE/(4*SUBSECTION_SIZE); stride > 0; stride /= 2) {
        __syncthreads();
        unsigned int subsection_index = (threadIdx.x + 1)*stride*2 - 1;
        if (subsection_index + stride < SECTION_SIZE/SUBSECTION_SIZE)
            shared_array[(subsection_index+stride+1)*SUBSECTION_SIZE-1] += shared_array[(subsection_index+1)*SUBSECTION_SIZE-1];
    }

    // Phase 3: add the last element of its predecessor's section to its element except for the last element.
    for (unsigned int index = 0; index < SUBSECTION_SIZE - 1; ++index) {
        if (threadIdx.x > 0)
            shared_array[global_start_index+index] += shared_array[global_start_index-1];
        if (global_start_index+index + blockDim.x*SUBSECTION_SIZE < size)
            shared_array[global_start_index+index + blockDim.x*SUBSECTION_SIZE] += shared_array[global_start_index + blockDim.x*SUBSECTION_SIZE-1];
        __syncthreads();
    }

    // Recall that two elements are stored at the end in Brent-Kung algorithm.
    for (unsigned int iter = 0; iter < SUBSECTION_SIZE*2; ++iter) {
        unsigned int index = iter*blockDim.x + threadIdx.x;
        if (index < size)
            output_array[index] = shared_array[index];
    }
}

void runParallelInclusiveScan(
    float* input_array_h,
    float* output_array_h,
    unsigned int size
) {
    // Get size in bytes.
    size_t size_input = size * sizeof(float);
    size_t size_output = size * sizeof(float);

    // Load and copy input_array and filter to device memory.
    float * input_array_d, * output_array_d;
    
    cudaMalloc((void***)&input_array_d, size_input);
    cudaMemcpy(input_array_d, input_array_h, size_input, cudaMemcpyHostToDevice);

    cudaMalloc((void***)&output_array_d, size_output);

    // Invoke kernel.
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(1);
    BrentKungScanWithThreadCoarseningKernel<<<dimGrid, dimBlock>>>(input_array_d, output_array_d, size);

    // Copy the output matrix from the device memory.
    cudaMemcpy(output_array_h, output_array_d, size_output, cudaMemcpyDeviceToHost);

    // Free device vectors.
    cudaFree(input_array_d);
    cudaFree(output_array_d);
}


int main() {
    unsigned int input_size = INPUT_SIZE;
    float input_array[input_size];
    float output_array[input_size];

    for (int i = 1; i < input_size + 1; ++i) 
        input_array[i-1] = i % 31;

    runParallelInclusiveScan(
        input_array,
        output_array,
        input_size
    );

    // Print the last 3 values of the output array.
    printf("%.0f %.0f %.0f \n", output_array[input_size - 3], output_array[input_size - 2], output_array[input_size - 1]);

    return 0;
}