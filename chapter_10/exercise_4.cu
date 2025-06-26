#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define INPUT_SIZE 10000
#define BLOCK_SIZE 1024
#define COARSE_FACTOR 2


/**
 * Enables atomicMax for floating point, taken from:
 * https://stackoverflow.com/questions/17399119/how-do-i-use-atomicmax-on-floating-point-values-in-cuda.
 */
__device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
    float old_value;
    old_value = !signbit(value) ? __int_as_float(atomicMax((int*)addr, __float_as_int(value))) :
        __uint_as_float(atomicMin((unsigned int*)addr, __float_as_uint(value)));

    return old_value;
}


/**
 *  Perform multiblock max reduction with thread coarsening, corresponds to Exercise 4.
 * 
 *  Note: it only works for input arrays with length of the power of two.
 */
__global__
void CoarsenedMaxReduction(
    float* input_array,
    float* output_value
) {
    __shared__ float input_shared[BLOCK_SIZE];
    // Each block processes COARSE_FACTOR * 2 * blockDim.x * COARSE_FACTOR;
    unsigned int segment = COARSE_FACTOR*2*blockDim.x*blockIdx.x;
    unsigned int index = segment + threadIdx.x;
    unsigned int shared_index = threadIdx.x;

    float max_value = input_array[index];
    for (unsigned int tile = 1; tile < COARSE_FACTOR*2; ++tile)
        max_value = max(max_value, input_array[index + tile*BLOCK_SIZE]);
    input_shared[shared_index] = max_value;

    for (unsigned int stride = blockDim.x/2; stride >= 1; stride/=2) {
        __syncthreads();
        if (shared_index < stride) 
            input_shared[shared_index] = max(input_shared[shared_index], input_shared[shared_index + stride]);
    }
    if (shared_index == 0) {
        atomicMaxFloat(output_value, input_shared[0]);
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
    CoarsenedMaxReduction<<<dimGrid, dimBlock>>>(input_array_d, output_value_d);

    // Copy the output matrix from the device memory.
    cudaMemcpy(output_value_h, output_value_d, size_output, cudaMemcpyDeviceToHost);

    // Free device vectors.
    cudaFree(input_array_d);
    cudaFree(output_value_d);
}

void swap (float *a, float *b) { 
    float temp = *a; 
    *a = *b; 
    *b = temp; 
} 

void shuffle(float *array, unsigned int array_size)  {
    srand(0); 

    for (int max_index = array_size - 1; max_index > 0; max_index--) 
    { 
        // Pick a random index from 0 to i.
        int picked_index = rand() % (max_index + 1); 
        swap(&array[max_index], &array[picked_index]); 
    } 
} 


int main() {
    unsigned int input_size = INPUT_SIZE;
    float input_array[input_size];
    float output_value;

    for (int i = 1; i < input_size + 1; ++i) 
        input_array[i-1] = i;
    
    shuffle(input_array, input_size);

    runSumReduction(
        input_array,
        &output_value
    );

    printf("%.0f\n", output_value);

    return 0;
}