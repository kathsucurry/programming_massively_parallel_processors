/**
 *  Perform sparse matrix-vector multiplication using ELL format, corresponds to chapter 14.4 & Exercise 4.
 *  The matrix-vector multiplication performs Y = AX where A denotes the matrix and X denotes the vector.
 */
#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define ELL_COL_NUM 2
#define MATRIX_DIM 8
#define THREADS_NUM_PER_BLOCK 4
#define eps 1e-5


struct EllCooSparseMatrix {
    unsigned int *ell_col_indices;
    float *ell_values;
    unsigned int *ell_num_nonzeroes_per_row;
    unsigned int ell_array_size;
    unsigned int *coo_row_indices;
    unsigned int *coo_col_indices;
    float *coo_values;
    unsigned int coo_array_size;
};


EllCooSparseMatrix convertCsrToEllCoo(
    unsigned int *csr_row_pointer_indices,
    unsigned int *csr_col_indices,
    float *csr_values,
    unsigned int predefined_max_row_elements
) {
    unsigned int *ell_num_nonzeroes_per_row = (unsigned int*)malloc(MATRIX_DIM * sizeof(unsigned int));
    unsigned int coo_num_elements = 0;

    // Obtain the number of nonzeros given the predefined max row elements.
    for (unsigned int row = 0; row < MATRIX_DIM; ++row) {
        unsigned int num_row_elements = csr_row_pointer_indices[row+1] - csr_row_pointer_indices[row];
        ell_num_nonzeroes_per_row[row] = min(predefined_max_row_elements, num_row_elements);
        if (num_row_elements > predefined_max_row_elements)
            coo_num_elements += num_row_elements - predefined_max_row_elements;
    }

    // Allocate memory for COO elements.
    unsigned int *coo_row_indices = (unsigned int*)malloc(coo_num_elements * sizeof(unsigned int));
    unsigned int *coo_col_indices = (unsigned int*)malloc(coo_num_elements * sizeof(unsigned int));
    float *coo_values = (float*)malloc(coo_num_elements * sizeof(float));
    unsigned int coo_element_index = 0;
    
    // Generate the padded tables.
    unsigned int padded_col_indices_table[MATRIX_DIM][predefined_max_row_elements];
    float padded_values_table[MATRIX_DIM][predefined_max_row_elements];
    for (unsigned int row = 0; row < MATRIX_DIM; ++row) {
        unsigned int init_num_row_elements = csr_row_pointer_indices[row+1] - csr_row_pointer_indices[row];
        // Handle ELL format.
        for (unsigned int col = 0; col < predefined_max_row_elements; ++col)
            if (col >= ell_num_nonzeroes_per_row[row]) {
                // Add padding.
                padded_col_indices_table[row][col] = 0;
                padded_values_table[row][col] = 0.0f;
            } else {
                padded_col_indices_table[row][col] = csr_col_indices[col + csr_row_pointer_indices[row]];
                padded_values_table[row][col] = csr_values[col + csr_row_pointer_indices[row]];
            }
        // Handle COO format.
        if (init_num_row_elements > predefined_max_row_elements) {
            for (unsigned int col = predefined_max_row_elements; col < init_num_row_elements; ++col) {
                coo_row_indices[coo_element_index] = row;
                coo_col_indices[coo_element_index] = csr_col_indices[col + csr_row_pointer_indices[row]];
                coo_values[coo_element_index] = csr_values[col + csr_row_pointer_indices[row]];
                ++coo_element_index;
            }
        }
    }
    
    size_t array_size = MATRIX_DIM * predefined_max_row_elements;
    unsigned int *ell_col_indices = (unsigned int *)malloc(array_size * sizeof(unsigned int));
    float *ell_values = (float *)malloc(array_size * sizeof(float));

    // Convert the table into column-major arrays.
    unsigned int new_array_index = 0;
    for (unsigned int col = 0; col < predefined_max_row_elements; ++col)
        for (unsigned int row = 0; row < MATRIX_DIM; ++row) {
            ell_col_indices[new_array_index] = padded_col_indices_table[row][col];
            ell_values[new_array_index] = padded_values_table[row][col];
            ++new_array_index;
        }
    
    EllCooSparseMatrix ellcoo_matrix;
    ellcoo_matrix.ell_col_indices = ell_col_indices;
    ellcoo_matrix.ell_values = ell_values;
    ellcoo_matrix.ell_num_nonzeroes_per_row = ell_num_nonzeroes_per_row;
    ellcoo_matrix.ell_array_size = MATRIX_DIM * predefined_max_row_elements;
    ellcoo_matrix.coo_row_indices = coo_row_indices;
    ellcoo_matrix.coo_col_indices = coo_col_indices;
    ellcoo_matrix.coo_values = coo_values;
    ellcoo_matrix.coo_array_size = coo_num_elements;
    return ellcoo_matrix;
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


/**
 * To perform SPMV, the function does the following steps:
 * 1) Launch the ELL kernel.
 * 2) Compute the contributions of the COO elements on the host.
 */
void runSparseMatVecMultiplication(
    EllCooSparseMatrix ellcoo_matrix,
    float* vector_X_h,
    float* vector_Y_h,
    unsigned int vector_dim
) {
    // STEP 1: LAUNCH ELL KERNEL.
    unsigned int *ell_col_indices_h = ellcoo_matrix.ell_col_indices;
    float *ell_values_h = ellcoo_matrix.ell_values;
    unsigned int *ell_num_nonzeroes_per_row_h = ellcoo_matrix.ell_num_nonzeroes_per_row;
    unsigned int ell_array_size = ellcoo_matrix.ell_array_size;

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

    // STEP 2: ADD THE CONTRIBUTION FROM COO ELEMENTS.
    unsigned int *coo_row_indices = ellcoo_matrix.coo_row_indices;
    unsigned int *coo_col_indices = ellcoo_matrix.coo_col_indices;
    float *coo_values = ellcoo_matrix.coo_values;
    for (int i = 0; i < ellcoo_matrix.coo_array_size; ++i) {
        vector_Y_h[coo_row_indices[i]] += (vector_X_h[coo_col_indices[i]] * coo_values[i]);
    }
}


int main() {
    // Convert CSR format to ELL-COO format for the following sparse matrix:
    // [ 1,  0,  0,  0,  0,  0,  0,  0]
    // [ 2,  3,  0,  4,  0,  5,  0,  6]
    // [ 0,  0,  7,  0,  8,  0,  0,  0]
    // [ 9,  0,  0, 10,  0,  0,  0,  0]
    // [ 0,  0,  0,  0, 11,  0,  0,  0]
    // [ 0, 12,  0,  0,  0, 13,  0,  0]
    // [14,  0, 15,  0, 16,  0, 17,  0]
    // [ 0,  0,  0,  0,  0,  0,  0, 18]
    unsigned int csr_row_pointer_indices[] = {0, 1, 6, 8, 10, 11, 13, 17, 18};
    unsigned int csr_col_indices[] = {0, 0, 1, 3, 5, 7, 2, 4, 0,  3,  4,  1,  5,  0,  2,  4,  6,  7};
    float csr_values[]             = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};

    // Prepare variables for ELL format; at this point we don't know the size of the
    // arrays yet until we know the max number of elements across rows.
    EllCooSparseMatrix ellcoo_matrix = convertCsrToEllCoo(
        csr_row_pointer_indices,
        csr_col_indices,
        csr_values,
        ELL_COL_NUM
    );

    // Prepare vector X for matrix-vector multiplication.
    float X[MATRIX_DIM];
    for (int i = 0; i < MATRIX_DIM; ++i)
        X[i] = 1;

    float Y_expected[] = {1, 20, 15, 19, 11, 25, 62, 18};
    float Y_actual[MATRIX_DIM];

    runSparseMatVecMultiplication(
        ellcoo_matrix,
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