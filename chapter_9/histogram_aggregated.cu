#include <stdio.h>
#include <cuda_runtime.h>


#define BLOCK_SIZE 16
// 26 alphabets divided into bins where each bin is for 4 consecutive alphabets.
#define NUM_BINS 7


/**
 *  Perform histogram with aggregation, corresponds to Fig. 9.15.
 */
__global__
void HistogramAggregatedKernel(
    char* data,
    unsigned int length,
    unsigned int *histogram
) {
    // Initialize privatized bins.
    __shared__ unsigned int histogram_shared[NUM_BINS];
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x)
        histogram_shared[bin] = 0u;
    __syncthreads();

    unsigned int accumulator = 0;
    // Tracks the index of the bin that was last encountered and is being aggregated.
    int prev_bin_index = -1;
    unsigned int index_start = blockIdx.x*blockDim.x + threadIdx.x;

    // blockDim.x * gridDim.x = the total number of threads.
    for (unsigned int index=index_start; index<length; index += blockDim.x*gridDim.x) {
        int alphabet_position = data[index] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26) {
            int bin = alphabet_position / 4;
            if (bin == prev_bin_index) {
                ++accumulator;
                continue;
            }
            
            if (accumulator > 0)
                atomicAdd(&(histogram_shared[prev_bin_index]), accumulator);
            accumulator = 1;
            prev_bin_index = bin;
        }
    }
    // Handle the existing accumulator.
    if (accumulator > 0)
        atomicAdd(&(histogram_shared[prev_bin_index]), accumulator);

    __syncthreads();

    // Store to the global memory, with the assumption where NUM_BINS < blockDIM.x.
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
    size_t size_histogram = NUM_BINS * sizeof(int);

    // Load and copy inputs to device memory.
    char *data_d;
    unsigned int *histogram_d;

    cudaMalloc((void***)&data_d, size_data);
    cudaMemcpy(data_d, data_h, size_data, cudaMemcpyHostToDevice);

    cudaMalloc((void***)&histogram_d, size_histogram);

    // Invoke kernel.
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(1);
    HistogramAggregatedKernel<<<dimGrid, dimBlock>>>(
        data_d,
        length,
        histogram_d
    );

    // Copy the output matrix from the device memory.
    cudaMemcpy(histogram_h, histogram_d, size_histogram, cudaMemcpyDeviceToHost);

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