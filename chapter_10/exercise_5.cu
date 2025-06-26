#include <stdio.h>
#include <cuda_runtime.h>

#define INPUT_SIZE 5891
#define BLOCK_SIZE 1024
#define COARSE_FACTOR 2


/**
 *  Perform multiblock sum reduction with thread coarsening for arbitrary input length,
 *  corresponds to Exercise 5.
 */
__global__
void CoarsenedSumReduction(
    float* input_array,
    float* output_value,
    unsigned int input_size
) {
    __shared__ float input_shared[BLOCK_SIZE];
    // Each block processes COARSE_FACTOR * 2 * blockDim.x * COARSE_FACTOR;
    unsigned int segment = COARSE_FACTOR*2*blockDim.x*blockIdx.x;
    unsigned int index = segment + threadIdx.x;
    unsigned int shared_index = threadIdx.x;

    if (index >= input_size)
        return;

    float sum = input_array[index];
    for (unsigned int tile = 1; tile < COARSE_FACTOR*2; ++tile) {
        if (index + tile*BLOCK_SIZE < input_size)
            sum += input_array[index + tile*BLOCK_SIZE];
    }
    input_shared[shared_index] = sum;

    for (unsigned int stride = blockDim.x/2; stride >= 1; stride/=2) {
        __syncthreads();
        if (shared_index < stride) 
            input_shared[shared_index] += input_shared[shared_index + stride];
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
    dim3 dimGrid(ceil(INPUT_SIZE / (2.0 * BLOCK_SIZE * COARSE_FACTOR)));
    CoarsenedSumReduction<<<dimGrid, dimBlock>>>(input_array_d, output_value_d, INPUT_SIZE);

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