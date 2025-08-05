/**
 *  Perform BFS with privatization of frontiers, corresponds to chapter 15.6 (Fig. 15.14).
 */
#include <stdio.h>
#include <limits.h>
#include <cuda_runtime.h>

#define VERTEX_LENGTH 9
#define LOCAL_FRONTIER_CAPACITY 2
#define THREADS_NUM_PER_BLOCK 4
#define BLOCK_NUM ceil(VERTEX_LENGTH * 1.0 / THREADS_NUM_PER_BLOCK)


struct CSRGraph {
    unsigned int *src_pointer_indices;
    unsigned int *dst_indices;
};


__global__
void BfsFrontierKernel(
    CSRGraph csr_graph,
    unsigned int *level,
    unsigned int *prev_frontier,
    unsigned int *curr_frontier,
    unsigned int num_prev_frontier,
    unsigned int *num_curr_frontier,
    unsigned int curr_level
) {
    // Initialize privatized frontier.
    __shared__ unsigned int shared_curr_frontier[LOCAL_FRONTIER_CAPACITY];
    __shared__ unsigned int shared_num_curr_frontier;
    if (threadIdx.x == 0)
        shared_num_curr_frontier = 0;
    __syncthreads();

    // Perform BFS.
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < num_prev_frontier) {
        unsigned int vertex = prev_frontier[index];
        for (unsigned int edge = csr_graph.src_pointer_indices[vertex];
                edge < csr_graph.src_pointer_indices[vertex + 1]; ++edge) {
            unsigned int neighbor = csr_graph.dst_indices[edge];
            if (atomicCAS(&level[neighbor], UINT_MAX, curr_level) == UINT_MAX) {
                unsigned int shared_curr_frontier_index = atomicAdd(&shared_num_curr_frontier, 1);
                if (shared_curr_frontier_index < LOCAL_FRONTIER_CAPACITY)
                    shared_curr_frontier[shared_curr_frontier_index] = neighbor;
                else {
                    // Recall that shared_num_curr_frontier is currently > LOCAL_FRONTIER_CAPACITY.
                    shared_num_curr_frontier = LOCAL_FRONTIER_CAPACITY;
                    unsigned int curr_frontier_index = atomicAdd(num_curr_frontier, 1);
                    curr_frontier[curr_frontier_index] = neighbor;
                }
            }
        }
    }
    __syncthreads();

    // Allocate in the global frontier.
    __shared__ unsigned int shared_curr_frontier_start_index;
    if (threadIdx.x == 0)
        shared_curr_frontier_start_index = atomicAdd(num_curr_frontier, shared_num_curr_frontier);
    __syncthreads();

    // Commit to the global frontier.
    for (unsigned int thread_frontier_index = threadIdx.x;
            thread_frontier_index < shared_num_curr_frontier;
            thread_frontier_index += blockDim.x) {
        unsigned int curr_frontier_index = shared_curr_frontier_start_index + thread_frontier_index;
        curr_frontier[curr_frontier_index] = shared_curr_frontier[thread_frontier_index];
    }
}


void runBFS(
    unsigned int *csr_src_pointer_indices_h,
    unsigned int *csr_dst_indices_h,
    unsigned int edge_length,
    unsigned int *level_h,
    unsigned int root
) {
    // Load and copy host variables to device memory.
    unsigned int *csr_src_pointer_indices_d, *csr_dst_indices_d, *level_d;
    // Frontier-related device variables.
    unsigned int *prev_frontier_d, *curr_frontier_d, *num_curr_frontier_d, *num_prev_frontier_d;

    size_t size_src_pointer_indices = (VERTEX_LENGTH + 1) * sizeof(unsigned int);
    size_t size_dst_indices = edge_length * sizeof(unsigned int);
    size_t size_vertex = VERTEX_LENGTH * sizeof(unsigned int);

    cudaMalloc((void**)&csr_src_pointer_indices_d, size_src_pointer_indices);
    cudaMemcpy(csr_src_pointer_indices_d, csr_src_pointer_indices_h, size_src_pointer_indices, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&csr_dst_indices_d, size_dst_indices);
    cudaMemcpy(csr_dst_indices_d, csr_dst_indices_h, size_dst_indices, cudaMemcpyHostToDevice);

    struct CSRGraph csr_graph_d = {
        .src_pointer_indices = csr_src_pointer_indices_d,
        .dst_indices = csr_dst_indices_d
    };

    cudaMalloc((void**)&level_d, size_vertex);
    // Initialize with "not visited" value.
    cudaMemset(level_d, UINT_MAX, size_vertex);

    cudaMalloc((void**)&prev_frontier_d, size_vertex);
    cudaMalloc((void**)&curr_frontier_d, size_vertex);
    cudaMalloc((void**)&num_curr_frontier_d, sizeof(unsigned int));
    cudaMalloc((void**)&num_prev_frontier_d, sizeof(unsigned int));

    // Invoke kernel per iteration.
    dim3 dimBlock(THREADS_NUM_PER_BLOCK);
    dim3 dimGrid(BLOCK_NUM);

    // Assign level[root] to 0 (i.e., visited).
    cudaMemset(level_d + root, 0, sizeof(unsigned int));
    unsigned int curr_level = 1;
    
    unsigned int *num_prev_frontier_h = (unsigned int*)malloc(sizeof(unsigned int));
    num_prev_frontier_h[0] = 1;
    cudaMemcpy(num_prev_frontier_d, num_prev_frontier_h, sizeof(unsigned int), cudaMemcpyHostToDevice);

    unsigned int *prev_frontier_h = (unsigned int*)malloc(VERTEX_LENGTH * sizeof(unsigned int));
    prev_frontier_h[0] = root;    
    cudaMemcpy(prev_frontier_d, prev_frontier_h, VERTEX_LENGTH * sizeof(unsigned int), cudaMemcpyHostToDevice);

    while (num_prev_frontier_h[0] != 0) {
        cudaMemset(num_curr_frontier_d, 0, sizeof(unsigned int));

        BfsFrontierKernel<<<dimGrid, dimBlock>>>(
            csr_graph_d,
            level_d,
            prev_frontier_d,
            curr_frontier_d,
            num_prev_frontier_h[0],
            num_curr_frontier_d,
            curr_level
        );
        cudaMemcpy(num_prev_frontier_h, num_curr_frontier_d, sizeof(unsigned int), cudaMemcpyDeviceToHost);

        // Switch prev and curr frontiers.
        unsigned int *temp = prev_frontier_d;
        prev_frontier_d = curr_frontier_d;
        curr_frontier_d = temp;

        temp = num_prev_frontier_d;
        num_prev_frontier_d = num_curr_frontier_d;
        num_curr_frontier_d = temp;

        ++curr_level;
    }

    // Copy level from the device memory.
    cudaMemcpy(level_h, level_d, size_vertex, cudaMemcpyDeviceToHost);

    free(num_prev_frontier_h);
    free(prev_frontier_h);

    // Free device arrays.
    cudaFree(csr_src_pointer_indices_d);
    cudaFree(csr_dst_indices_d);
    cudaFree(level_d);
    cudaFree(prev_frontier_d);
    cudaFree(curr_frontier_d);
    cudaFree(num_prev_frontier_d);
    cudaFree(num_curr_frontier_d);
}


int main() {
    // Generate the CSR graph; all values (edge weight) are 1s.
    unsigned int src_pointer_indices[] = {0, 2, 4, 7, 9, 11, 12, 13, 15, 15};
    unsigned int dst_indices[] = {1, 2, 3, 4, 5, 6, 7, 4, 8, 5, 8, 6, 8, 0, 6};
    unsigned int edge_length = 15;

    unsigned int root = 2;
    unsigned int *level = (unsigned int*)malloc(VERTEX_LENGTH * sizeof(unsigned int));
    runBFS(
        src_pointer_indices,
        dst_indices,
        edge_length,
        level,
        root
    );

    for (unsigned int i = 0; i < VERTEX_LENGTH; ++i)
        printf("%u ", level[i]);
    printf("\n");

    return 0;
}