#include <stdio.h>
#include <cuda_runtime.h>

// BLOCK_SIZE and TILE_WIDTH should be identical.
#define BLOCK_SIZE 4
#define TILE_WIDTH 4


/**
 *  Performs a simple matrix multiplication with shared memory for matrices with any size
 *  (as long as the sizes are valid for performing matrix multiplication).
 */
__global__
void MatrixMultiplicationKernel(
    int* matrix_M,
    int* matrix_N,
    int* matrix_Out,
    int mat1_Height,
    int mat1_Width,
    int mat2_Width
) {
    // Initialize the shared memories.
    __shared__  float shared_M[TILE_WIDTH][TILE_WIDTH];
    __shared__  float shared_N[TILE_WIDTH][TILE_WIDTH];

    int block_x = blockIdx.x;
    int block_y = blockIdx.y;
    int thread_x = threadIdx.x;
    int thread_y = threadIdx.y;

    // Identify the row and column of the matrix_Out matrix to work on.
    int Row = block_y * TILE_WIDTH + thread_y;
    int Col = block_x * TILE_WIDTH + thread_x;

    // Loop over the tiles required to compute matrix_Out elements.
    int out_value = 0;
    for (int phase = 0; phase < ceil(mat1_Width*1.0/TILE_WIDTH); ++phase) {
        // Collaboratively load M and N tiles into shared memory.
        int in_matrix_M_index = Row*mat1_Width + phase*TILE_WIDTH + thread_x;
        int in_matrix_N_index = (phase*TILE_WIDTH + thread_y)*mat2_Width + Col;

        // Check for boundary for both dimensions.
        if (Row < mat1_Height && phase*TILE_WIDTH + thread_x < mat1_Width)
            shared_M[thread_y][thread_x] = matrix_M[in_matrix_M_index];
        else
            shared_M[thread_y][thread_x] = 0;

        if (phase*TILE_WIDTH + thread_y < mat1_Width && Col < mat2_Width)
            shared_N[thread_y][thread_x] = matrix_N[in_matrix_N_index];
        else
            shared_N[thread_y][thread_x] = 0;
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            out_value += shared_M[thread_y][k] * shared_N[k][thread_x];
        }
        __syncthreads();
    }

    if (Row < mat1_Height && Col < mat2_Width)
        matrix_Out[Row * mat2_Width + Col] = out_value;
 }


void runMatrixMultiplication(
    int* matrix_M_h,
    int* matrix_N_h,
    int* matrix_Out_h,
    int mat1_Height,
    int mat1_Width,
    int mat2_Width
) {
    // Get size in bytes.
    size_t size_M = mat1_Height * mat1_Width * sizeof(int);
    size_t size_N = mat1_Width * mat2_Width * sizeof(int);
    size_t size_Out = mat1_Height * mat2_Width * sizeof(int);

    // Load and copy matrix M and N to device memory.
    int * matrix_M_d, * matrix_N_d, * matrix_Out_d;
    cudaMalloc((void***)&matrix_M_d, size_M);
    cudaMemcpy(matrix_M_d, matrix_M_h, size_M, cudaMemcpyHostToDevice);

    cudaMalloc((void***)&matrix_N_d, size_N);
    cudaMemcpy(matrix_N_d, matrix_N_h, size_N, cudaMemcpyHostToDevice);

    cudaMalloc((void***)&matrix_Out_d, size_Out);

    // Invoke kernel.
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(ceil(mat2_Width / (BLOCK_SIZE * 1.0)), ceil(mat1_Height / (BLOCK_SIZE * 1.0)));
    MatrixMultiplicationKernel<<<dimGrid, dimBlock>>>(
        matrix_M_d,
        matrix_N_d,
        matrix_Out_d,
        mat1_Height,
        mat1_Width,
        mat2_Width
    );

    // Copy the output matrix from the device memory.
    cudaMemcpy(matrix_Out_h, matrix_Out_d, size_Out, cudaMemcpyDeviceToHost);

    // Free device vectors.
    cudaFree(matrix_M_d);
    cudaFree(matrix_N_d);
    cudaFree(matrix_Out_d);
}


void fillMatrix(int * matrix, int height, int width) {
    for (int i = 0; i < height * width; ++i) {
        matrix[i] = i + 1;
    }
}


int main() {
    // Matrices are stored in row-major order.
    // Define the sizes of the matrices:
    // - matrix M [mat1_Height, mat1_Width],
    // - matrix N [mat1_Width, mat2_Width] (m2_height = m1_width).
    int mat1_Height = 10;
    int mat1_Width = 9;
    int mat2_Width = 15;

    // Define matrices M and N where each element = 1 .. height * width.
    int * matrix_M = (int *) malloc(mat1_Height * mat1_Width * sizeof(int));
    int * matrix_N = (int *) malloc(mat1_Width * mat2_Width * sizeof(int));
    int * matrix_Out = (int *) malloc(mat1_Height * mat2_Width * sizeof(int));

    fillMatrix(matrix_M, mat1_Height, mat1_Width);
    fillMatrix(matrix_N, mat1_Width, mat2_Width);

    runMatrixMultiplication(matrix_M, matrix_N, matrix_Out, mat1_Height, mat1_Width, mat2_Width);

    for (int i = 0; i < mat1_Height; ++i) {
        for (int j = 0; j < mat2_Width; ++j) {
            printf("%d ", matrix_Out[i * mat2_Width + j]);
        }
        printf("\n");
    }

    free(matrix_M);
    free(matrix_N);
    free(matrix_Out);

    return 0;
}