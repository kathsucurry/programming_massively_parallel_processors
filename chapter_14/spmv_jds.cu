/**
 *  Perform sparse matrix-vector multiplication using JDS format, corresponds to chapter 14.6 & Exercise 5.
 *  The matrix-vector multiplication performs Y = AX where A denotes the matrix and X denotes the vector.
 */
#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#include <algorithm>

#define ELL_COL_NUM 2
#define MATRIX_DIM 8
#define THREADS_NUM_PER_BLOCK 4
#define eps 1e-5


struct JdsSparseMatrix {
    unsigned int *iter_pointer_indices;
    unsigned int *col_indices;
    unsigned int *row_order;
    float *values;
    unsigned int size_array;
};


unsigned int * getArgSort(const unsigned int *values, const unsigned int size) {
    // Generate the index array.
    unsigned int *indices = (unsigned int*)malloc(size * sizeof(unsigned int));
    for (unsigned int i = 0; i < size; ++i)
        indices[i] = i;
    
    std::stable_sort(
        indices,
        indices + size,
        [values](unsigned int index1, unsigned int index2) {
            return values[index1] > values[index2];
        }
    );

    return indices;
}


JdsSparseMatrix convertCsrToJDS(
    unsigned int *csr_row_pointer_indices,
    unsigned int *csr_col_indices,
    float *csr_values
) {
    unsigned int num_nonzeroes_per_row[MATRIX_DIM];

    // Obtain the max number of elements across each row.
    unsigned int max_row_elements = 0;
    unsigned int total_num_non_zeroes = 0;
    for (unsigned int row = 0; row < MATRIX_DIM; ++row) {
        unsigned int num_row_elements = csr_row_pointer_indices[row+1] - csr_row_pointer_indices[row];
        max_row_elements = max(max_row_elements, num_row_elements);
        num_nonzeroes_per_row[row] = num_row_elements;
        total_num_non_zeroes += num_row_elements;
    }
    
    // Generate the padded tables.
    unsigned int padded_col_indices_table[MATRIX_DIM][max_row_elements];
    float padded_values_table[MATRIX_DIM][max_row_elements];
    for (unsigned int row = 0; row < MATRIX_DIM; ++row) {
        for (unsigned int col = 0; col < max_row_elements; ++col)
            if (col >= num_nonzeroes_per_row[row]) {
                padded_col_indices_table[row][col] = 0;
                padded_values_table[row][col] = 0.0f;
            } else {
                padded_col_indices_table[row][col] = csr_col_indices[col + csr_row_pointer_indices[row]];
                padded_values_table[row][col] = csr_values[col + csr_row_pointer_indices[row]];
            }
    }

    // Get the indices of the sorted array by descending size.
    unsigned int *jds_row_order = getArgSort(num_nonzeroes_per_row, MATRIX_DIM);

    // Convert the table into column-major arrays.
    unsigned int *jds_iter_pointer_indices = (unsigned int*)malloc(
        (max_row_elements + 1) * sizeof(unsigned int));
    unsigned int *jds_col_indices = (unsigned int*)malloc(total_num_non_zeroes * sizeof(unsigned int));
    float *jds_values = (float*)malloc(total_num_non_zeroes * sizeof(float));
    
    unsigned int array_counter = 0;
    unsigned int iter_pointer_counter = 0;
    for (unsigned int col = 0; col < max_row_elements; ++col)
        for (unsigned int row = 0; row < MATRIX_DIM; ++row) {
            if (row == 0) {
                jds_iter_pointer_indices[iter_pointer_counter++] = array_counter;
            }
            unsigned int original_row_index = jds_row_order[row];
            if (col >= num_nonzeroes_per_row[original_row_index])
                break;
            jds_col_indices[array_counter] = padded_col_indices_table[original_row_index][col];
            jds_values[array_counter] = padded_values_table[original_row_index][col];
            ++array_counter;
        }
    
    // Add the starting location of the nonexistent row.
    jds_iter_pointer_indices[iter_pointer_counter] = array_counter;

    for (int i = 0; i < total_num_non_zeroes; ++i) {
        printf("%d ", jds_col_indices[i]);
    }
    printf("\n");

    for (int i = 0; i < total_num_non_zeroes; ++i) {
        printf("%.0f ", jds_values[i]);
    }
    printf("\n");

    JdsSparseMatrix jds_matrix;
    jds_matrix.iter_pointer_indices = jds_iter_pointer_indices;
    jds_matrix.col_indices = jds_col_indices;
    jds_matrix.row_order = jds_row_order;
    jds_matrix.values = jds_values;
    jds_matrix.size_array = total_num_non_zeroes;

    return jds_matrix;
}


// __global__
// void SparseMatVecMulEllKernel(
//     unsigned int* col_indices,
//     float* values,
//     unsigned int* num_nonzeroes_per_row,
//     unsigned int array_size,
//     float* vector_X,
//     float* vector_Y
// ) {
//     unsigned int row_index = blockIdx.x * blockDim.x + threadIdx.x;
//     if (row_index < MATRIX_DIM) {
//         float sum = 0.0f;
//         for (unsigned int col = 0; col < num_nonzeroes_per_row[row_index]; ++col) {
//             unsigned int index = col * MATRIX_DIM + row_index;
//             unsigned int col_index = col_indices[index];
//             float value = values[index];
//             sum += vector_X[col_index] * value;
//         } 
//         vector_Y[row_index] = sum;
//     }
// }


// /**
//  * To perform SPMV, the function does the following steps:
//  * 1) Launch the ELL kernel.
//  * 2) Compute the contributions of the COO elements on the host.
//  */
// void runSparseMatVecMultiplication(
//     EllCooSparseMatrix ellcoo_matrix,
//     float* vector_X_h,
//     float* vector_Y_h,
//     unsigned int vector_dim
// ) {
//     // STEP 1: LAUNCH ELL KERNEL.
//     unsigned int *ell_col_indices_h = ellcoo_matrix.ell_col_indices;
//     float *ell_values_h = ellcoo_matrix.ell_values;
//     unsigned int *ell_num_nonzeroes_per_row_h = ellcoo_matrix.ell_num_nonzeroes_per_row;
//     unsigned int ell_array_size = ellcoo_matrix.ell_array_size;

//     // Load and copy host variables to device memory.
//     // Load the matrix.
//     unsigned int *ell_col_indices_d, *ell_num_nonzeroes_per_row_d;
//     float *ell_values_d;

//     size_t size_col_indices = ell_array_size * sizeof(unsigned int);
//     size_t size_values = ell_array_size * sizeof(float);
//     size_t size_num_nonzeroes_per_row = MATRIX_DIM * sizeof(unsigned int);

//     cudaMalloc((void**)&ell_col_indices_d, size_col_indices);
//     cudaMemcpy(ell_col_indices_d, ell_col_indices_h, size_col_indices, cudaMemcpyHostToDevice);

//     cudaMalloc((void**)&ell_values_d, size_values);
//     cudaMemcpy(ell_values_d, ell_values_h, size_values, cudaMemcpyHostToDevice);

//     cudaMalloc((void**)&ell_num_nonzeroes_per_row_d, size_num_nonzeroes_per_row);
//     cudaMemcpy(ell_num_nonzeroes_per_row_d, ell_num_nonzeroes_per_row_h, size_num_nonzeroes_per_row, cudaMemcpyHostToDevice);

//     // Load vector X and Y.
//     float *vector_X_d, *vector_Y_d;
//     size_t size_array = vector_dim * sizeof(float);
//     cudaMalloc((void***)&vector_X_d, size_array);
//     cudaMemcpy(vector_X_d, vector_X_h, size_array, cudaMemcpyHostToDevice);
//     cudaMalloc((void***)&vector_Y_d, size_array);

//     // Invoke kernel per iteration.
//     dim3 dimBlock(THREADS_NUM_PER_BLOCK);
//     dim3 dimGrid(ceil(MATRIX_DIM * 1.0 / THREADS_NUM_PER_BLOCK));
    
//     SparseMatVecMulEllKernel<<<dimGrid, dimBlock>>>(
//         ell_col_indices_d, ell_values_d, ell_num_nonzeroes_per_row_d, ell_array_size, vector_X_d, vector_Y_d);

//     // Copy the output from the device memory.
//     cudaMemcpy(vector_Y_h, vector_Y_d, size_array, cudaMemcpyDeviceToHost);

//     // Free device arrays.
//     cudaFree(vector_X_d);
//     cudaFree(vector_Y_d);
//     cudaFree(ell_col_indices_d);
//     cudaFree(ell_values_d);
//     cudaFree(ell_num_nonzeroes_per_row_d);

//     // STEP 2: ADD THE CONTRIBUTION FROM COO ELEMENTS.
//     unsigned int *coo_row_indices = ellcoo_matrix.coo_row_indices;
//     unsigned int *coo_col_indices = ellcoo_matrix.coo_col_indices;
//     float *coo_values = ellcoo_matrix.coo_values;
//     for (int i = 0; i < ellcoo_matrix.coo_array_size; ++i) {
//         vector_Y_h[coo_row_indices[i]] += (vector_X_h[coo_col_indices[i]] * coo_values[i]);
//     }
// }


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

    JdsSparseMatrix jds_matrix = convertCsrToJDS(
        csr_row_pointer_indices,
        csr_col_indices,
        csr_values
    );

    // // Prepare vector X for matrix-vector multiplication.
    // float X[MATRIX_DIM];
    // for (int i = 0; i < MATRIX_DIM; ++i)
    //     X[i] = 1;

    // float Y_expected[] = {1, 20, 15, 19, 11, 25, 62, 18};
    // float Y_actual[MATRIX_DIM];

    // runSparseMatVecMultiplication(
    //     ellcoo_matrix,
    //     X,
    //     Y_actual,
    //     MATRIX_DIM
    // );

    // // Check if the result is correct.
    // bool is_correct = true;
    // for (int i = 0; i < MATRIX_DIM; ++i)
    //     if (fabs(Y_actual[i] - Y_expected[i]) > eps) {
    //         is_correct = false;
    //         printf("The actual and expected results differ at index %d; actual = %.0f, expected = %.0f\n", i, Y_actual[i], Y_expected[i]);
    //         break;
    //     }
    // if (is_correct)
    //     printf("The actual and expected results are identical!\n");
    
    return 0;
}