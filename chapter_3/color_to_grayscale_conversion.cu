#include <stdint.h>
#include <cuda_runtime.h>

// Image-related libraries taken from https://github.com/nothings/stb.
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


const int NUM_RGB_CHANNELS = 3;
const int NUM_GS_CHANNELS = 1;


__global__
void colorToGrayscaleConversionKernel(
    // The input image is encoded as unsigned chars [0, 255].
    // Each pixel is 3 consecutive chars for the 3 channels (RGB).
    unsigned char * Pout,
    unsigned char * Pin,
    int width,
    int height
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        // Get 1D offset for the grayscale image.
        int grayOffset = row * width + col;

        // One can think of the RGB image having CHANNEL times
        // more columns than the grayscale image.
        int rgbOffset = grayOffset * NUM_RGB_CHANNELS;

        unsigned char r = Pin[rgbOffset    ];
        unsigned char g = Pin[rgbOffset + 1];
        unsigned char b = Pin[rgbOffset + 2];

        // Perform the rescaling and store it.
        // We multiply by floating point constants.
        Pout[grayOffset] = 0.21f * r + 0.71f * g + 0.07f * b;
    }
}


int main() {
    int width, height, num_channels;

    // Assume valid image; ideally should handle invalid image.
    // Input image name currently hardcoded. One improvement is to enable command line options.
    unsigned char * rgb_image_h = stbi_load("input.jpg", &width, &height, &num_channels, NUM_RGB_CHANNELS);

    // Calculate the size in RGB and grayscale.
    int rgb_size = height * width * NUM_RGB_CHANNELS * sizeof(rgb_image_h[0]);
    int gs_size = height * width * NUM_GS_CHANNELS * sizeof(rgb_image_h[0]);
    
    // Initialize and allocate memory for the host grayscale image.
    unsigned char * rgb_image_d, * gs_image_d;
    unsigned char * gs_image_h = (unsigned char *)malloc(gs_size);

    // Allocate device memory.
    cudaMalloc((void***)&rgb_image_d, rgb_size);
    cudaMalloc((void***)&gs_image_d, gs_size);

    // Copy rgb_image and gs_image from host to device memory.
    cudaMemcpy(rgb_image_d, rgb_image_h, rgb_size, cudaMemcpyHostToDevice);
    stbi_image_free(rgb_image_h);

    // Call the kernel function.
    // Set a block with 16 threads (16 = arbitrarily selected).
    dim3 dimGrid(ceil(width/16.0), ceil(height/16.0), 1);
    dim3 dimBlock(16, 16, 1);
    colorToGrayscaleConversionKernel<<<dimGrid, dimBlock>>>(gs_image_d, rgb_image_d, width, height);

    // Copy the grayscale image output from the device memory.
    cudaMemcpy(gs_image_h, gs_image_d, gs_size, cudaMemcpyDeviceToHost);

    // Free device variables.
    cudaFree(rgb_image_d);
    cudaFree(gs_image_d);

    // Generate the output image; currently hardcoded name.
    stbi_write_jpg("output.jpg", width, height, NUM_GS_CHANNELS, gs_image_h, 100);
    stbi_image_free(gs_image_h);

    return 0;
}