#include <stdio.h>
#include <cuda_runtime.h>


#define BLOCK_SIZE 16
// 26 alphabets divided into bins where each bin is for 4 consecutive alphabets.
#define NUM_BINS 7


/**
 *  Perform histogram with privatization using shared memory, corresponds to Fig. 9.10.
 */
__global__
void HistogramPrivateKernel(
    char* data,
    unsigned int length,
    unsigned int *histogram
) {
    // Initialize privatized bins.
    __shared__ unsigned int histogram_shared[NUM_BINS];
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x)
        histogram_shared[bin] = 0;
    __syncthreads();

    unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index < length) {
        int character_hist_location = data[index] - 'a';
        // Ensure valid alphabet.
        if (character_hist_location >= 0 && character_hist_location < 26)
            // Recall that shared memory can only be accessed by the threads from the same block.
            atomicAdd(&(histogram_shared[character_hist_location/4]), 1);
    }
    __syncthreads();

    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        unsigned int bin_value = histogram_shared[bin];
        if (bin_value > 0)
            atomicAdd(&(histogram[bin]), bin_value);
    }
}


void generateHistogram(
    char *data_h,
    unsigned int length,
    unsigned int *histogram_h
) {
    // Get size in bytes.
    size_t size_data = length * sizeof(char);

    // Ensure it has enough allocation for the private copy as well.
    unsigned int dim_grid = ceil(length * 1.0 / BLOCK_SIZE);

    size_t size_histogram_with_private = dim_grid*NUM_BINS * sizeof(int);
    size_t size_histogram_original = NUM_BINS * sizeof(int);

    // Load and copy inputs to device memory.
    char *data_d;
    unsigned int *histogram_d;

    cudaMalloc((void***)&data_d, size_data);
    cudaMemcpy(data_d, data_h, size_data, cudaMemcpyHostToDevice);

    cudaMalloc((void***)&histogram_d, size_histogram_with_private);

    // Invoke kernel.
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(ceil(length / (BLOCK_SIZE * 1.0)));
    HistogramPrivateKernel<<<dimGrid, dimBlock>>>(
        data_d,
        length,
        histogram_d
    );

    // Copy the output matrix from the device memory.
    cudaMemcpy(histogram_h, histogram_d, size_histogram_original, cudaMemcpyDeviceToHost);

    // Free device vectors.
    cudaFree(data_d);
    cudaFree(histogram_d);
}


int main() {
    // Define the input characters.
    char input[] = "programming massively parallel processors";
    unsigned int length_input = strlen(input); // 41.

    unsigned int length_histogram = NUM_BINS; 

    unsigned int histogram[length_histogram] = {0};

    generateHistogram(
        input,
        length_input,
        histogram
    );

    for (int i = 0; i < length_histogram; ++i) {
        printf("%d %d\n", i, histogram[i]);
    }

    return 0;
}