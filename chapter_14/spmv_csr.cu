/**
 *  Perform sparse matrix-vector multiplication using compressed sparse row (csr) format, corresponds to chapter 14.3.
 *  The matrix-vector multiplication performs Y = AX where A denotes the matrix and X denotes the vector.
 */
#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define INPUT_LENGTH 8
#define MATRIX_DIM 4
#define THREADS_NUM_PER_BLOCK 4
#define BLOCK_NUM ceil(INPUT_LENGTH * 1.0 / THREADS_NUM_PER_BLOCK)
#define eps 1e-5


__global__
void SparseMatVecMulCsrKernel(
    unsigned int* csr_matrix_row_pointer_indices,
    unsigned int* csr_matrix_col_indices,
    float* csr_matrix_values,
    unsigned int csr_matrix_nonzeroes_size,
    float* vector_X,
    float* vector_Y
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < MATRIX_DIM) {
        float sum = 0.0f;
        for (unsigned int i = csr_matrix_row_pointer_indices[row]; i < csr_matrix_row_pointer_indices[row + 1]; ++i) {
            unsigned int col = csr_matrix_col_indices[i];
            float value = csr_matrix_values[i];
            sum += vector_X[col] * value;
        }
        vector_Y[row] = sum;
    }
}


void runSparseMatVecMultiplication(
    unsigned int* row_pointer_indices_h,
    unsigned int* col_indices_h,
    float* values_h,
    unsigned int num_nonzeroes,
    float* vector_X_h,
    float* vector_Y_h,
    unsigned int vector_dim
) {
    // Load and copy host variables to device memory.
    float *vector_X_d, *vector_Y_d;
    size_t size_array = vector_dim * sizeof(float);
    cudaMalloc((void***)&vector_X_d, size_array);
    cudaMemcpy(vector_X_d, vector_X_h, size_array, cudaMemcpyHostToDevice);
    cudaMalloc((void***)&vector_Y_d, size_array);

    unsigned int *row_pointer_indices_d, *col_indices_d;
    float *values_d;
    // Recall that the row pointer array includes one nonexistent row at the end.
    size_t size_array_pointers = (MATRIX_DIM + 1) * sizeof(unsigned int);
    size_t size_array_indices = num_nonzeroes * sizeof(unsigned int);
    size_t size_array_values = num_nonzeroes * sizeof(float);
    
    cudaMalloc((void**)&row_pointer_indices_d, size_array_pointers);
    cudaMemcpy(row_pointer_indices_d, row_pointer_indices_h, size_array_pointers, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&col_indices_d, size_array_indices);
    cudaMemcpy(col_indices_d, col_indices_h, size_array_indices, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&values_d, size_array_values);
    cudaMemcpy(values_d, values_h, size_array_values, cudaMemcpyHostToDevice);

    // Invoke kernel per iteration.
    dim3 dimBlock(THREADS_NUM_PER_BLOCK);
    dim3 dimGrid(BLOCK_NUM);
    
    SparseMatVecMulCsrKernel<<<dimGrid, dimBlock>>>(
        row_pointer_indices_d, col_indices_d, values_d, num_nonzeroes, vector_X_d, vector_Y_d);

    // Copy the output from the device memory.
    cudaMemcpy(vector_Y_h, vector_Y_d, size_array, cudaMemcpyDeviceToHost);

    // Free device arrays.
    cudaFree(vector_X_d);
    cudaFree(vector_Y_d);
    cudaFree(row_pointer_indices_d);
    cudaFree(col_indices_d);
    cudaFree(values_d);
}


int main() {
    // Generate the CSR matrix.
    unsigned int row_pointer_indices[] = {0, 2, 5, 7, 8};
    unsigned int col_indices[] = {0, 1, 0, 2, 3, 1, 2, 3};
    float values[] = {1, 7, 5, 3, 9, 2, 8, 6};

    // Prepare vector X for matrix-vector multiplication.
    float X[] = {1, 2, 3, 4};
    float Y_expected[] = {15, 50, 28, 24};
    float Y_actual[MATRIX_DIM];

    runSparseMatVecMultiplication(
        row_pointer_indices,
        col_indices,
        values,
        INPUT_LENGTH,
        X,
        Y_actual,
        MATRIX_DIM
    );

    // Check if the result is correct.
    bool is_correct = true;
    for (int i = 0; i < MATRIX_DIM; ++i)
        if (fabs(Y_actual[i] - Y_expected[i]) > eps) {
            is_correct = false;
            printf("The actual and expected results differ at index %d; actual = %.0f, expected = %.0f\n", i, Y_actual[i], Y_expected[i]);
            break;
        }
    if (is_correct)
        printf("The actual and expected results are identical!\n");

    return 0;
}