#include <stdio.h>
#include <cuda_runtime.h>


#define BLOCK_SIZE 16


/**
 *  Perform basic histogram, corresponds to Fig. 9.6.
 */
__global__
void HistogramKernel(
    char* data,
    unsigned int length,
    unsigned int *histogram
) {
    unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index < length) {
        int character_hist_location = data[index] - 'a';
        // Ensure valid alphabet.
        if (character_hist_location >= 0 && character_hist_location < 26)
            atomicAdd(&(histogram[character_hist_location/4]), 1);
    }
}


void generateHistogram(
    char *data_h,
    unsigned int length,
    unsigned int *histogram_h
) {
    // Get size in bytes.
    size_t size_data = length * sizeof(char);
    // 26 alphabets divided into bins where each bin is for 4 consecutive alphabets.
    size_t size_histogram = ceil(26.0 / 4) * sizeof(int);

    // Load and copy inputs to device memory.
    char *data_d;
    unsigned int *histogram_d;

    cudaMalloc((void***)&data_d, size_data);
    cudaMemcpy(data_d, data_h, size_data, cudaMemcpyHostToDevice);

    cudaMalloc((void***)&histogram_d, size_histogram);

    // Invoke kernel.
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(ceil(length / (BLOCK_SIZE * 1.0)));
    HistogramKernel<<<dimGrid, dimBlock>>>(
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
    unsigned int length_input = strlen(input);

    unsigned int length_histogram = 7; 

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