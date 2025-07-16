#include <stdio.h>
#include <cuda_runtime.h>

#define SECTION_SIZE 32
#define BLOCK_SIZE SECTION_SIZE
// For now, assume that the input size is the same as the block size.
#define INPUT_SIZE SECTION_SIZE


/**
 *  Perform Kogge-Stone kernel for inclusive (segmented) scan, corresponds to Fig. 11.3 with double-buffering implemented.
 */
__global__
void KoggeStoneScanDoubleBufferKernel(
    float* input_array,
    float* output_array,
    unsigned int size
) {
    __shared__ float buffer_1_array[SECTION_SIZE];
    __shared__ float buffer_2_array[SECTION_SIZE];
    unsigned int global_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_index < size)
        buffer_1_array[threadIdx.x] = input_array[global_index];
    else
        buffer_1_array[threadIdx.x] = 0.0f;

    // Store the iteration index to determine which buffer should be read or written.
    unsigned int iter_index = 0;
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();

        if (threadIdx.x >= stride) {
            if (iter_index % 2 == 0)
                buffer_2_array[threadIdx.x] = buffer_1_array[threadIdx.x] + buffer_1_array[threadIdx.x - stride];
            else
                buffer_1_array[threadIdx.x] = buffer_2_array[threadIdx.x] + buffer_2_array[threadIdx.x - stride];
        }

        ++iter_index;
    }

    if (global_index < size) {
        if (iter_index % 2 == 0)
            output_array[global_index] = buffer_1_array[threadIdx.x];
        else
            output_array[global_index] = buffer_2_array[threadIdx.x];
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
    dim3 dimGrid(ceil(size / BLOCK_SIZE));
    KoggeStoneScanDoubleBufferKernel<<<dimGrid, dimBlock>>>(input_array_d, output_array_d, size);

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