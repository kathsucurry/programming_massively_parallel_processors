#include <stdio.h>
#include <cuda_runtime.h>


#define BLOCK_SIZE 16
// 26 alphabets divided into bins where each bin is for 4 consecutive alphabets.
#define NUM_BINS 7


/**
 *  Perform histogram with privatization, corresponds to Fig. 9.9.
 */
__global__
void HistogramPrivateKernel(
    char* data,
    unsigned int length,
    unsigned int *histogram
) {
    unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index < length) {
        int character_hist_location = data[index] - 'a';
        // Ensure valid alphabet.
        if (character_hist_location >= 0 && character_hist_location < 26)
            // Enough device memory should be allocated to hold all private copies of the histogram,
            // where the private copies of the histogram = gridDim.x * NUM_BINS * 4B.
            // The offset shifts the position to the private copy for the block the thread belongs.
            // The goal is to reduce contention.
            atomicAdd(&(histogram[blockIdx.x*NUM_BINS + character_hist_location/4]), 1);
    }

    if (blockIdx.x > 0) {
        __syncthreads();

        // Run computation only when thread index is 0 since it already iterates through
        // all the bins of the block.
        if (threadIdx.x > 0)
            return;

        // Add all the values in private copy into the version produced by block 0.
        // Note that we want to iterate from one thread to the next block of that thread
        for (unsigned int bin=0; bin<NUM_BINS; bin += 1) {
            unsigned int bin_value = histogram[blockIdx.x*NUM_BINS + bin];
            if (bin_value > 0)
                atomicAdd(&(histogram[bin]), bin_value);
        }
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