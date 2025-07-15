#include <stdio.h>
#include <cuda_runtime.h>

#define SECTION_SIZE 2048
#define BLOCK_SIZE SECTION_SIZE / 2
#define INPUT_SIZE SECTION_SIZE


/**
 *  Perform Brent-Kung kernel for inclusive (segmented) scan, corresponds to Fig. 11.7.
 */
__global__
void BrentKungScanKernel(
    float* input_array,
    float* output_array,
    unsigned int size
) {
    __shared__ float shared_array[SECTION_SIZE];
    unsigned int global_index = 2*blockIdx.x*blockDim.x + threadIdx.x;
    if (global_index < size)
        shared_array[threadIdx.x] = input_array[global_index];
    
    // Recall that SECTION_SIZE can have up to 2*BLOCK_SIZE.
    if (global_index + blockDim.x < size)
        shared_array[threadIdx.x + blockDim.x] = input_array[global_index + blockDim.x];
    
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        __syncthreads();
        unsigned int index = (threadIdx.x + 1)*2*stride - 1;
        if (index < SECTION_SIZE)
            shared_array[index] += shared_array[index - stride];
    }

    for (int stride = SECTION_SIZE/4; stride > 0; stride /= 2) {
        __syncthreads();
        unsigned int index = (threadIdx.x + 1)*stride*2 - 1;
        if (index + stride < SECTION_SIZE)
            shared_array[index + stride] += shared_array[index];
    }

    __syncthreads();
    if (global_index < size)
        output_array[global_index] = shared_array[threadIdx.x];
    
    if (global_index + blockDim.x < size)
        output_array[global_index + blockDim.x] = shared_array[threadIdx.x + blockDim.x];
}

void runParallelInclusiveScan(
    float* input_array_h,
    float* output_value_h,
    unsigned int size
) {
    // Get size in bytes.
    size_t size_input = size * sizeof(float);
    size_t size_output = size * sizeof(float);

    // Load and copy input_array and filter to device memory.
    float * input_array_d, * output_value_d;
    
    cudaMalloc((void***)&input_array_d, size_input);
    cudaMemcpy(input_array_d, input_array_h, size_input, cudaMemcpyHostToDevice);

    cudaMalloc((void***)&output_value_d, size_output);

    // Invoke kernel.
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(ceil(size / (2.0 * BLOCK_SIZE)));
    BrentKungScanKernel<<<dimGrid, dimBlock>>>(input_array_d, output_value_d, size);

    // Copy the output matrix from the device memory.
    cudaMemcpy(output_value_h, output_value_d, size_output, cudaMemcpyDeviceToHost);

    // Free device vectors.
    cudaFree(input_array_d);
    cudaFree(output_value_d);
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