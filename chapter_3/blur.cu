#include <stdint.h>
#include <cuda_runtime.h>

// Image-related libraries taken from https://github.com/nothings/stb.
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


const int NUM_RGB_CHANNELS = 3;
const int BLUR_SIZE = 6;


/**
 * Blurs an RGB image given BLUR_SIZE.
 * 
 * The function is a modified version of the function from the book
 * to enable blurring an RGB image instead of a grayscale image.
 * 
 */
__global__
void blurKernelRGB(
    unsigned char * image_input,
    unsigned char * image_output,
    int width,
    int height
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        // Blur the image = computing the average of the surrounding BLUR_SIZE x BLUR_SIZE box.
        // Define the variables for getting the average.
        int rgb_pixel_value_total[] = {0, 0, 0};
        // One pixel count suffices since all count values for each RGB are the same.
        int pixel_count = 0;
        int rgb_offset = (row * width + col) * NUM_RGB_CHANNELS;

        for (int blur_row = -BLUR_SIZE; blur_row < BLUR_SIZE + 1; ++blur_row) {
            for (int blur_col = -BLUR_SIZE; blur_col < BLUR_SIZE + 1; ++blur_col) {
                int current_row = row + blur_row;
                int current_col = col + blur_col;

                // Ensure current_row and current_col are valid image pixel indices.
                // Check the first value in RGB should suffice.
                if (current_row >= 0 && current_row < height && current_col >= 0 && current_col < width) {
                    int current_rgb_offset = (current_row * width + current_col) * NUM_RGB_CHANNELS;
                    pixel_count += 1;
                    for (int rgb_index = 0; rgb_index < NUM_RGB_CHANNELS; ++rgb_index) {
                        rgb_pixel_value_total[rgb_index] += image_input[current_rgb_offset + rgb_index];
                    }
                }
            }
        }

        // Update the output.
        for (int rgb_index = 0; rgb_index < NUM_RGB_CHANNELS; ++rgb_index) {
            image_output[rgb_offset + rgb_index] = (unsigned char)(rgb_pixel_value_total[rgb_index] / pixel_count);
        }
    }
}


int main() {
    int width, height, num_channels;

    // Assume valid image; ideally should handle invalid image.
    // Input image name currently hardcoded. One improvement is to enable command line options.
    unsigned char * image_input_h = stbi_load("input.jpg", &width, &height, &num_channels, NUM_RGB_CHANNELS);

    // Calculate the size and allocate memory for output image.
    int image_size = height * width * NUM_RGB_CHANNELS * sizeof(image_input_h[0]);
    unsigned char * image_output_h = (unsigned char *)malloc(image_size);
    
    // Initialize and allocate memory for the device variables.
    unsigned char * image_input_d, * image_output_d;
    cudaMalloc((void***)&image_input_d, image_size);
    cudaMalloc((void***)&image_output_d, image_size);

    // Copy image_input from host to device memory.
    cudaMemcpy(image_input_d, image_input_h, image_size, cudaMemcpyHostToDevice);
    stbi_image_free(image_input_h);

    // Call the kernel function.
    // Set a block with 16 threads (16 = arbitrarily selected).
    dim3 dimGrid(ceil(width/16.0), ceil(height/16.0), 1);
    dim3 dimBlock(16, 16, 1);
    blurKernelRGB<<<dimGrid, dimBlock>>>(image_input_d, image_output_d, width, height);

    // Copy the image output from the device memory.
    cudaMemcpy(image_output_h, image_output_d, image_size, cudaMemcpyDeviceToHost);

    // Free device variables.
    cudaFree(image_input_d);
    cudaFree(image_output_d);

    // Generate the output image; currently hardcoded name.
    stbi_write_jpg("output.jpg", width, height, NUM_RGB_CHANNELS, image_output_h, 100);
    stbi_image_free(image_output_h);

    return 0;
}