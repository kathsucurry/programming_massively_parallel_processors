/**
 *  Perform sparse matrix-vector multiplication using ELL format, corresponds to chapter 14.4.
 *  The matrix-vector multiplication performs Y = AX where A denotes the matrix and X denotes the vector.
 */
#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define MATRIX_DIM 4
#define THREADS_NUM_PER_BLOCK 4
#define eps 1e-5


struct EllSparseMatrix {
    unsigned int *col_indices;
    float *values;
    unsigned int *num_nonzeroes_per_row;
    unsigned int array_size;
};


EllSparseMatrix convertCsrToEll(
    unsigned int *csr_row_pointer_indices,
    unsigned int *csr_col_indices,
    float *csr_values
) {
    unsigned int ell_num_nonzeroes_per_row[MATRIX_DIM];

    // Obtain the max number of elements across each row.
    unsigned int max_row_elements = 0;
    for (unsigned int row = 0; row < MATRIX_DIM; ++row) {
        unsigned int num_row_elements = csr_row_pointer_indices[row+1] - csr_row_pointer_indices[row];
        max_row_elements = max(max_row_elements, num_row_elements);
        ell_num_nonzeroes_per_row[row] = num_row_elements;
    }
    
    // Generate the padded tables.
    unsigned int padded_col_indices_table[MATRIX_DIM][max_row_elements];
    float padded_values_table[MATRIX_DIM][max_row_elements];
    for (unsigned int row = 0; row < MATRIX_DIM; ++row) {
        for (unsigned int col = 0; col < max_row_elements; ++col)
            if (col >= ell_num_nonzeroes_per_row[row]) {
                padded_col_indices_table[row][col] = 0;
                padded_values_table[row][col] = 0.0f;
            } else {
                padded_col_indices_table[row][col] = csr_col_indices[col + csr_row_pointer_indices[row]];
                padded_values_table[row][col] = csr_values[col + csr_row_pointer_indices[row]];
            }
    }
    
    size_t array_size = MATRIX_DIM * max_row_elements;
    unsigned int *ell_col_indices = (unsigned int *)malloc(array_size * sizeof(unsigned int));
    float *ell_values = (float *)malloc(array_size * sizeof(float));

    // Convert the table into column-major arrays.
    unsigned int new_array_index = 0;
    for (unsigned int col = 0; col < max_row_elements; ++col)
        for (unsigned int row = 0; row < MATRIX_DIM; ++row) {
            ell_col_indices[new_array_index] = padded_col_indices_table[row][col];
            ell_values[new_array_index] = padded_values_table[row][col];
            ++new_array_index;
        }
    
    EllSparseMatrix ell_matrix;
    ell_matrix.col_indices = ell_col_indices;
    ell_matrix.values = ell_values;
    ell_matrix.num_nonzeroes_per_row = ell_num_nonzeroes_per_row;
    ell_matrix.array_size = MATRIX_DIM * max_row_elements;
    return ell_matrix;
}


__global__
void SparseMatVecMulEllKernel(
    unsigned int* col_indices,
    float* values,
    unsigned int* num_nonzeroes_per_row,
    unsigned int array_size,
    float* vector_X,
    float* vector_Y
) {
    unsigned int row_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_index < MATRIX_DIM) {
        float sum = 0.0f;
        for (unsigned int col = 0; col < num_nonzeroes_per_row[row_index]; ++col) {
            unsigned int index = col * MATRIX_DIM + row_index;
            unsigned int col_index = col_indices[index];
            float value = values[index];
            sum += vector_X[col_index] * value;
        } 
        vector_Y[row_index] = sum;
    }
}


void runSparseMatVecMultiplication(
    EllSparseMatrix ell_matrix,
    float* vector_X_h,
    float* vector_Y_h,
    unsigned int vector_dim
) {
    unsigned int *ell_col_indices_h = ell_matrix.col_indices;
    float *ell_values_h = ell_matrix.values;
    unsigned int *ell_num_nonzeroes_per_row_h = ell_matrix.num_nonzeroes_per_row;
    unsigned int ell_array_size = ell_matrix.array_size;

    // Load and copy host variables to device memory.
    // Load the matrix.
    unsigned int *ell_col_indices_d, *ell_num_nonzeroes_per_row_d;
    float *ell_values_d;

    size_t size_col_indices = ell_array_size * sizeof(unsigned int);
    size_t size_values = ell_array_size * sizeof(float);
    size_t size_num_nonzeroes_per_row = MATRIX_DIM * sizeof(unsigned int);

    cudaMalloc((void**)&ell_col_indices_d, size_col_indices);
    cudaMemcpy(ell_col_indices_d, ell_col_indices_h, size_col_indices, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&ell_values_d, size_values);
    cudaMemcpy(ell_values_d, ell_values_h, size_values, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&ell_num_nonzeroes_per_row_d, size_num_nonzeroes_per_row);
    cudaMemcpy(ell_num_nonzeroes_per_row_d, ell_num_nonzeroes_per_row_h, size_num_nonzeroes_per_row, cudaMemcpyHostToDevice);

    // Load vector X and Y.
    float *vector_X_d, *vector_Y_d;
    size_t size_array = vector_dim * sizeof(float);
    cudaMalloc((void***)&vector_X_d, size_array);
    cudaMemcpy(vector_X_d, vector_X_h, size_array, cudaMemcpyHostToDevice);
    cudaMalloc((void***)&vector_Y_d, size_array);

    // Invoke kernel per iteration.
    dim3 dimBlock(THREADS_NUM_PER_BLOCK);
    dim3 dimGrid(ceil(MATRIX_DIM * 1.0 / THREADS_NUM_PER_BLOCK));
    
    SparseMatVecMulEllKernel<<<dimGrid, dimBlock>>>(
        ell_col_indices_d, ell_values_d, ell_num_nonzeroes_per_row_d, ell_array_size, vector_X_d, vector_Y_d);

    // Copy the output from the device memory.
    cudaMemcpy(vector_Y_h, vector_Y_d, size_array, cudaMemcpyDeviceToHost);

    // Free device arrays.
    cudaFree(vector_X_d);
    cudaFree(vector_Y_d);
    cudaFree(ell_col_indices_d);
    cudaFree(ell_values_d);
    cudaFree(ell_num_nonzeroes_per_row_d);
}


int main() {
    // Prepare a CSR matrix, which will then be converted into ELL format.
    unsigned int csr_row_pointer_indices[] = {0, 2, 5, 7, 8};
    unsigned int csr_col_indices[] = {0, 1, 0, 2, 3, 1, 2, 3};
    float csr_values[] = {1, 7, 5, 3, 9, 2, 8, 6};

    // Prepare variables for ELL format; at this point we don't know the size of the
    // arrays yet until we know the max number of elements across rows.
    EllSparseMatrix ell_matrix = convertCsrToEll(
        csr_row_pointer_indices,
        csr_col_indices,
        csr_values
    );

    // Prepare vector X for matrix-vector multiplication.
    float X[] = {1, 2, 3, 4};
    float Y_expected[] = {15, 50, 28, 24};
    float Y_actual[MATRIX_DIM];

    runSparseMatVecMultiplication(
        ell_matrix,
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