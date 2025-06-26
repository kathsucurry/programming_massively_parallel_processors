#include <stdio.h>
#include <cuda_runtime.h>


/**
 *  Perform sum reduction similar to Fig. 10.9 using the access pattern from exercise 3.
 */
__global__
void SumReductionExercise3(
    float* input_array,
    float* output_value,
    unsigned int input_size
) {
    unsigned int index = 2 * blockDim.x - threadIdx.x - 1;
    for (unsigned int stride = blockDim.x; stride >= 1; stride/=2) {
        if (threadIdx.x < stride) {
            input_array[index] += input_array[index - stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        *output_value = input_array[2 * blockDim.x - 1];
    }
}

void runSumReduction(
    float* input_array_h,
    float* output_value_h,
    unsigned int input_size
) {
    // Get size in bytes.
    size_t size_input = input_size * sizeof(float);
    size_t size_output = sizeof(float);

    // Load and copy input_array and filter to device memory.
    float * input_array_d, * output_value_d;
    
    cudaMalloc((void***)&input_array_d, size_input);
    cudaMemcpy(input_array_d, input_array_h, size_input, cudaMemcpyHostToDevice);

    cudaMalloc((void***)&output_value_d, size_output);

    unsigned int block_size = input_size / 2;

    // Invoke kernel.
    dim3 dimBlock(block_size);
    dim3 dimGrid(1);
    SumReductionExercise3<<<dimGrid, dimBlock>>>(input_array_d, output_value_d, input_size);

    // Copy the output matrix from the device memory.
    cudaMemcpy(output_value_h, output_value_d, size_output, cudaMemcpyDeviceToHost);

    // Free device vectors.
    cudaFree(input_array_d);
    cudaFree(output_value_d);
}


int main() {
    unsigned int input_size = 1024;
    float input_array[input_size];
    float output_value;

    for (int i = 1; i < input_size + 1; ++i) 
        input_array[i-1] = i;

    runSumReduction(
        input_array,
        &output_value,
        input_size
    );

    printf("%.0f\n", output_value);

    return 0;
}