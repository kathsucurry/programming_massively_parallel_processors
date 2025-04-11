/**
 * Question:
 * Implement a matrix multiplication kernel on same-width square matrices where:
 * a. each thread produces one output matrix row.
 * b. each thread produces one output matrix column.
 * 
 * Please refer to the README.md in the Exercises directory for
 * further descriptions of what I am trying to do in the code.
 */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

// Thread block size.
#define BLOCK_SIZE 16

typedef void (MatrixMultiplicationFuction)(int*, int*, int*, int);


/**
 * Each thread produces one output matrix row.
 */
__global__
void Question1AKernel(
    int* matrix_M,
    int* matrix_N,
    int* matrix_Out,
    int Width
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < Width) {
        for (int col = 0; col < Width; ++col) {
            int out_value = 0;
            for (int k = 0; k < Width; ++k) {
                out_value += matrix_M[row * Width + k] * matrix_N[Width * k + col];
            }
            matrix_Out[row * Width + col] = out_value;
        }
    }
}


/**
 * Each thread produces one output column row.
 */
__global__
void Question1BKernel(
    int* matrix_M,
    int* matrix_N,
    int* matrix_Out,
    int Width
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < Width) {
        for (int row = 0; row < Width; ++row) {
            int out_value = 0;
            for (int k = 0; k < Width; ++k) {
                out_value += matrix_M[row * Width + k] * matrix_N[Width * k + col];
            }
            matrix_Out[row * Width + col] = out_value;
        }
    }
}


/**
 * The host function, to deal with memory allocations and kernel function calls.
 */
void runMatrixMultiplication(
    int* matrix_M_h,
    int* matrix_N_h,
    int* matrix_Out_h,
    int Width,
    MatrixMultiplicationFuction* matmul_kernel
) {
    // Get size in bytes.
    size_t size = Width * Width * sizeof(int);

    // Load and copy matrix M and N to device memory.
    int * matrix_M_d, * matrix_N_d, * matrix_Out_d;
    cudaMalloc((void***)&matrix_M_d, size);
    cudaMemcpy(matrix_M_d, matrix_M_h, size, cudaMemcpyHostToDevice);

    cudaMalloc((void***)&matrix_N_d, size);
    cudaMemcpy(matrix_N_d, matrix_N_h, size, cudaMemcpyHostToDevice);

    cudaMalloc((void***)&matrix_Out_d, size);

    // Invoke kernel.
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(ceil(Width / (BLOCK_SIZE * 0.1)));
    matmul_kernel<<<dimGrid, dimBlock>>>(matrix_M_d, matrix_N_d, matrix_Out_d, Width);

    // Copy the output matrix from the device memory.
    cudaMemcpy(matrix_Out_h, matrix_Out_d, size, cudaMemcpyDeviceToHost);

    // Free device vectors.
    cudaFree(matrix_M_d);
    cudaFree(matrix_N_d);
    cudaFree(matrix_Out_d);
}


void parse_line_to_matrix(char * line, int * matrix, int matrix_width) {
    const int INIT_VALUE = -1;
    int matrix_index = 0, current_value = INIT_VALUE, char_index = 0;
    while (true) {
        if (line[char_index] == ' ' || line[char_index] == '\0' || line[char_index] == '\n') {
            if (current_value != INIT_VALUE) {
                matrix[matrix_index] = current_value;
                current_value = INIT_VALUE;
                ++matrix_index;
                ++char_index;
                continue;
            }

            if (line[char_index] == '\0' || line[char_index] == '\n') {
                break;
            }
        }

        int value = line[char_index] - '0';
        if (current_value == INIT_VALUE) {
            current_value = value;
        } else {
            current_value = current_value * 10 + value;
        }
        ++char_index;
    }
    assert (matrix_width * matrix_width == matrix_index);
}



/**
 * Reads a square matrix from file and returns the matrix width.
 *
 * The file contains 2 lines:
 * - Line 1: an integer indicating the width of the matrix.
 * - Line 2: the [width x width] element of matrix (all rows appended into one row) where each
 *           value is separated by a space. The line always ends with an end of line.
 *
 */
int read_matrix_from_file(int ** matrix, size_t max_buffer_size, const char* filename) {
    FILE* file_pointer = fopen(filename, "r");

    if (file_pointer == NULL) {
        fprintf(stderr, "Failed to open file with name %s\n", filename);
        return -1;
    }

    // Allocate memory for the line buffer.
    char *buffer = (char *) malloc(max_buffer_size);

    // Initialize the line counter;
    int line_counter = 0;
    int matrix_width = NULL;
    while (fgets(buffer, max_buffer_size, file_pointer)) {
        if (line_counter == 0) {
            sscanf(buffer, "%d\n", &matrix_width);
            * matrix = (int *) malloc(matrix_width * sizeof(int));
        } else if (line_counter == 1) {
            parse_line_to_matrix(buffer, * matrix, matrix_width);
        }
        ++line_counter;
    }
    free(buffer);
    fclose(file_pointer);

    return matrix_width;
}


int main() {
    // Matrices are to be stored in row-major order.

    int * matrix_M;
    int * matrix_N;
    int * matrix_Out;
    int * matrix_expected_Out;

    // Read inputs; we already know the approximate length of the characters.
    size_t input_max_length = 50000, output_max_length = 110000;

    // (matrix_M, input_max_length, "chapter_3_matmul_input_sample.txt");
    int matrix_width_M = read_matrix_from_file(&matrix_M, input_max_length, "chapter_3_matmul_input_sample.txt");
    int matrix_width_N = read_matrix_from_file(&matrix_N, input_max_length, "chapter_3_matmul_input_sample.txt");
    int matrix_width_expected_Out = read_matrix_from_file(&matrix_expected_Out, input_max_length, "chapter_3_matmul_input_sample.txt");

    assert (matrix_width_M == matrix_width_N && matrix_width_N == matrix_width_expected_Out);
    matrix_Out = (int *) malloc(matrix_width_expected_Out * sizeof(int));
    int Width = matrix_width_M;

    // Pick either question 1a or question 1b kernel for testing purposes.
    MatrixMultiplicationFuction* kernel_function = Question1BKernel;
    runMatrixMultiplication(matrix_M, matrix_N, matrix_Out, Width, kernel_function);

    // TODO: Use the real data and add runtime calculation.
    for (int i = 0; i < Width; ++i) {
        for (int j = 0; j < Width; ++j) {
            printf("%d ", matrix_Out[i * Width + j]);
        }
        printf("\n");
    }

    return 0;
}