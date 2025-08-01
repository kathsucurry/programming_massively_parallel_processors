/**
 *  Convert COO to CSR format, corresponds to chapter 14.3. SPMV is used for evaluating the CSR outputs.
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


__device__
void runSinglePassInlusiveScan(
    unsigned int* row_histogram,
    unsigned int* block_scan_values,
    unsigned int* block_flags,
    unsigned int N
) {
    __shared__ unsigned int shared_array[THREADS_NUM_PER_BLOCK];
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

    unsigned int shared_block_id = blockIdx.x;

    /* STAGE 1: PERFORM SCAN THROUGHOUT THE BLOCK/SECTION */
    // Phase 1: transfer data from global to shared memory.
    if (index < N)
        shared_array[threadIdx.x] = row_histogram[index];
    else
        shared_array[threadIdx.x] = 0.0f;
    __syncthreads();

    // Phase 2: perform scan operation on the elements.
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        float temp_value;
        if (threadIdx.x >= stride)
            temp_value = shared_array[threadIdx.x] + shared_array[threadIdx.x-stride];
        __syncthreads();

        if (threadIdx.x >= stride)
            shared_array[threadIdx.x] = temp_value;
        __syncthreads();
    }

    // Store the last scan value per block.
    if (threadIdx.x == blockDim.x - 1) {
        block_scan_values[shared_block_id] = shared_array[threadIdx.x];
        if (shared_block_id == 0) {
            // Start the domino from block_id 0.
            atomicAdd(&(block_flags[shared_block_id]), 1);
        }
    }

    // Ensure all block_scan_values have been updated.
    __syncthreads();

    /* STAGE 2: OBTAIN SUM VALUE FROM PREDECESSOR BLOCK */
    // Store the accumulated sum from the previous block.
    __shared__ float shared_previous_sum;
    if (threadIdx.x == 0 && shared_block_id > 0) {
        // Wait for the previous flag; recall that atomicAdd returns the original value.
        while (atomicAdd(&block_flags[shared_block_id - 1], 0) == 0) {}

        // Read the previous partial sum.
        shared_previous_sum = block_scan_values[shared_block_id - 1];
        
        // Propagate the partial sum.
        block_scan_values[shared_block_id] += shared_previous_sum;

        // Memory fence.
        __threadfence();

        // Update the flag.
        atomicAdd(&block_flags[shared_block_id], 1);
    } else {
        shared_previous_sum = 0;
    }
    __syncthreads();

    if (threadIdx.x < N)
        row_histogram[index] = shared_array[threadIdx.x] + shared_previous_sum;
}


__global__
void CooToCsrConversionKernel(
    unsigned int* coo_row_indices,
    unsigned int* coo_col_indices,
    float* coo_values,
    unsigned int* csr_row_pointer_indices,
    unsigned int* csr_col_indices,
    float* csr_values,
    unsigned int num_nonzeroes,
    unsigned int* row_histogram,
    unsigned int* block_row_scan_values,
    unsigned int* block_flags,
    unsigned int* block_gate
) {
    unsigned int global_index = blockIdx.x * blockDim.x + threadIdx.x;

    // Perform histogram to obtain the number of nonzeros elements per row.
    // Note that the index of the col index/value within the bin is recorded.
    unsigned int index_given_row;
    if (global_index < num_nonzeroes) {
        unsigned int row_index = coo_row_indices[global_index];
        index_given_row = atomicAdd(&row_histogram[row_index], 1);
    }
    __syncthreads();

    // Perform inclusive scan on the row histogram.
    runSinglePassInlusiveScan(row_histogram, block_row_scan_values, block_flags, MATRIX_DIM);

    // Store the scan results to global memory, converting inclusive to exclusive scan.
    if (global_index <= MATRIX_DIM)
        if (global_index == 0)
            csr_row_pointer_indices[global_index] = 0;
        else
            csr_row_pointer_indices[global_index] = row_histogram[global_index - 1];
    __syncthreads();

    // Ensure all blocks have updated the row pointer indices.
    atomicAdd(&block_gate[0], 1);
    while (atomicAdd(&block_gate[0], 0) < gridDim.x) {}

    // Reorder col and values accordingly.
    if (global_index < num_nonzeroes) {
        unsigned int destination = index_given_row + csr_row_pointer_indices[coo_row_indices[global_index]];
        csr_col_indices[destination] = coo_col_indices[global_index];
        csr_values[destination] = coo_values[global_index];
    }
}


void convertCooToCsr(
    unsigned int* coo_row_indices_h,
    unsigned int* coo_col_indices_h,
    float* coo_values_h,
    unsigned int* csr_row_pointer_indices_h,
    unsigned int* csr_col_indices_h,
    float* csr_values_h,
    unsigned int num_nonzeroes
) {
    // Load and copy host variables to device memory.
    unsigned int *coo_row_indices_d, *coo_col_indices_d;
    unsigned int *csr_row_pointer_indices_d, *csr_col_indices_d;
    float *coo_values_d, *csr_values_d;

    size_t size_indices = num_nonzeroes * sizeof(unsigned int);
    size_t size_values = num_nonzeroes * sizeof(float);
    size_t size_row_pointers = (MATRIX_DIM + 1) * sizeof(unsigned int);

    cudaMalloc((void**)&coo_row_indices_d, size_indices);
    cudaMemcpy(coo_row_indices_d, coo_row_indices_h, size_indices, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&coo_col_indices_d, size_indices);
    cudaMemcpy(coo_col_indices_d, coo_col_indices_h, size_indices, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&coo_values_d, size_indices);
    cudaMemcpy(coo_values_d, coo_values_h, size_values, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&csr_row_pointer_indices_d, size_row_pointers);
    cudaMalloc((void**)&csr_col_indices_d, size_indices);
    cudaMalloc((void**)&csr_values_d, size_values);

    // Prepare variables for supporting the conversion.
    unsigned int *row_histogram_d, *block_row_scan_values_d, *block_flags_d, *block_gate_d;
    cudaMalloc((void**)&row_histogram_d, MATRIX_DIM * sizeof(unsigned int));
    cudaMalloc((void**)&block_row_scan_values_d, BLOCK_NUM * sizeof(unsigned int));
    cudaMalloc((void**)&block_flags_d, BLOCK_NUM * sizeof(unsigned int));
    cudaMalloc((void**)&block_gate_d, sizeof(unsigned int));

    // Invoke kernel per iteration.
    dim3 dimBlock(THREADS_NUM_PER_BLOCK);
    dim3 dimGrid(BLOCK_NUM);
    
    CooToCsrConversionKernel<<<dimGrid, dimBlock>>>(
        coo_row_indices_d,
        coo_col_indices_d,
        coo_values_d,
        csr_row_pointer_indices_d,
        csr_col_indices_d,
        csr_values_d,
        num_nonzeroes,
        row_histogram_d,
        block_row_scan_values_d,
        block_flags_d,
        block_gate_d
    );

    // Copy the output from the device memory.
    cudaMemcpy(csr_row_pointer_indices_h, csr_row_pointer_indices_d, size_row_pointers, cudaMemcpyDeviceToHost);
    cudaMemcpy(csr_col_indices_h, csr_col_indices_d, size_indices, cudaMemcpyDeviceToHost);
    cudaMemcpy(csr_values_h, csr_values_d, size_values, cudaMemcpyDeviceToHost);

    // Free device arrays.
    cudaFree(coo_row_indices_d);
    cudaFree(coo_col_indices_d);
    cudaFree(coo_values_d);
    cudaFree(csr_row_pointer_indices_d);
    cudaFree(csr_col_indices_d);
    cudaFree(csr_values_d);
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
    // Prepare the COO matrix: we shouldn't assume that it's sorted by row.
    unsigned int coo_row_indices[] = {1, 3, 2, 2, 0, 1, 1, 0};
    unsigned int coo_col_indices[] = {3, 3, 1, 2, 0, 0, 2, 1};
    float coo_values[]             = {9, 6, 2, 8, 1, 5, 3, 7};

    // Prepare the CSR matrix outputs.
    unsigned int csr_row_pointer_indices[MATRIX_DIM + 1];
    unsigned int csr_col_indices[INPUT_LENGTH];
    float csr_values[INPUT_LENGTH];

    // Prepare vector X for matrix-vector multiplication.
    float X[] = {1, 2, 3, 4};
    float Y_expected[] = {15, 50, 28, 24};
    float Y_actual[MATRIX_DIM];

    convertCooToCsr(
        coo_row_indices,
        coo_col_indices,
        coo_values,
        csr_row_pointer_indices,
        csr_col_indices,
        csr_values,
        INPUT_LENGTH
    );

    // Run SPMV for testing purposes.
    runSparseMatVecMultiplication(
        csr_row_pointer_indices,
        csr_col_indices,
        csr_values,
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