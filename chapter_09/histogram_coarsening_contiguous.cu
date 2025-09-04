#include <stdio.h>
#include <cuda_runtime.h>


#define BLOCK_SIZE 16
// 26 alphabets divided into bins where each bin is for 4 consecutive alphabets.
#define NUM_BINS 7
#define COARSE_FACTOR 3


/**
 *  Perform histogram with coarsening using contiguous partitioning, corresponds to Fig. 9.12.
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
        histogram_shared[bin] = 0u;
    __syncthreads();

    // index_offset below refers to the index of the segment (of elements).
    unsigned int index_offset = blockIdx.x*blockDim.x + threadIdx.x;
    for (unsigned int index=index_offset*COARSE_FACTOR; index<min((index_offset+1)*COARSE_FACTOR, length); ++index) {
        int character_hist_location = data[index] - 'a';
        // Ensure valid alphabet.
        if (character_hist_location >= 0 && character_hist_location < 26)
            // Recall that shared memory can only be accessed by the threads from the same block.
            atomicAdd(&(histogram_shared[character_hist_location/4]), 1);
    }
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
    dim3 dimGrid(ceil(length * 1.0 / BLOCK_SIZE / COARSE_FACTOR));
    HistogramPrivateKernel<<<dimGrid, dimBlock>>>(
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