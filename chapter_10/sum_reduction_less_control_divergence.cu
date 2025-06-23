#include <stdio.h>
#include <cuda_runtime.h>


/**
 *  Perform sum reduction with less control divergence, corresponds to Fig. 10.9.
 */
__global__
void SumReductionLessControlDivergence(
    float* input_array,
    float* output_value,
    unsigned int input_size
) {
    unsigned int index = threadIdx.x;
    // For handling odd size values.
    unsigned int previous_stride = input_size;
    for (unsigned int stride = blockDim.x; stride >= 1; stride = ceil(stride / 2.0)) {
        if (threadIdx.x + stride < previous_stride) {
            input_array[index] += input_array[index + stride];
        }
        // Since ceil(stride / 2.0) = 1, we need to ensure that the loops eventually ends.
        if (stride == 1)
            break;
        previous_stride = stride;
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        *output_value = input_array[0];
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

    unsigned int block_size = ceil(input_size / 2.0);

    // Invoke kernel.
    dim3 dimBlock(block_size);
    dim3 dimGrid(1);
    SumReductionLessControlDivergence<<<dimGrid, dimBlock>>>(input_array_d, output_value_d, input_size);

    // Copy the output matrix from the device memory.
    cudaMemcpy(output_value_h, output_value_d, size_output, cudaMemcpyDeviceToHost);

    // Free device vectors.
    cudaFree(input_array_d);
    cudaFree(output_value_d);
}


int main() {
    unsigned int input_size = 20;
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