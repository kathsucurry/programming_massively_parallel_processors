#include <stdio.h>
#include <cuda_runtime.h>

#define INPUT_SIZE 5000
#define BLOCK_SIZE 1024


/**
 *  Perform multiblock sum reduction, corresponds to Fig. 10.13.
 */
__global__
void SegmentedSumReduction(
    float* input_array,
    float* output_value
) {
    __shared__ float input_shared[BLOCK_SIZE];
    // Each block processes 2 * blockDim.x;
    unsigned int segment = 2*blockDim.x*blockIdx.x;
    unsigned int index = segment + threadIdx.x;
    unsigned int shared_index = threadIdx.x;

    input_shared[shared_index] = input_array[index] + input_array[index + BLOCK_SIZE];

    // For handling odd size values.
    unsigned int previous_stride = BLOCK_SIZE;
    for (unsigned int stride = ceil(blockDim.x / 2.0); stride >= 1; stride = ceil(stride / 2.0)) {
        __syncthreads();
        if (threadIdx.x + stride < previous_stride) {
            input_shared[shared_index] += input_shared[shared_index + stride];
        }
        // Since ceil(stride / 2.0) = 1, we need to ensure that the loops eventually ends.
        if (stride == 1)
            break;
        previous_stride = stride;
    }
    if (shared_index == 0) {
        atomicAdd(output_value, input_shared[0]);
    }
}

void runSumReduction(
    float* input_array_h,
    float* output_value_h
) {
    // Get size in bytes.
    size_t size_input = INPUT_SIZE * sizeof(float);
    size_t size_output = sizeof(float);

    // Load and copy input_array and filter to device memory.
    float * input_array_d, * output_value_d;
    
    cudaMalloc((void***)&input_array_d, size_input);
    cudaMemcpy(input_array_d, input_array_h, size_input, cudaMemcpyHostToDevice);

    cudaMalloc((void***)&output_value_d, size_output);

    // Invoke kernel.
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(ceil(INPUT_SIZE * 1.0 / BLOCK_SIZE));
    SegmentedSumReduction<<<dimGrid, dimBlock>>>(input_array_d, output_value_d);

    // Copy the output matrix from the device memory.
    cudaMemcpy(output_value_h, output_value_d, size_output, cudaMemcpyDeviceToHost);

    // Free device vectors.
    cudaFree(input_array_d);
    cudaFree(output_value_d);
}


int main() {
    unsigned int input_size = INPUT_SIZE;
    float input_array[input_size];
    float output_value;

    for (int i = 1; i < input_size + 1; ++i) 
        input_array[i-1] = i;

    runSumReduction(
        input_array,
        &output_value
    );

    printf("%.0f\n", output_value);

    return 0;
}