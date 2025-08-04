/**
 *  Perform BFS top down, corresponds to chapter 15.3 (Fig. 15.6).
 */
#include <stdio.h>
#include <limits.h>
#include <cuda_runtime.h>

#define VERTEX_LENGTH 9
#define THREADS_NUM_PER_BLOCK 4
#define BLOCK_NUM ceil(VERTEX_LENGTH * 1.0 / THREADS_NUM_PER_BLOCK)


__global__
void BfsTopDownKernel(
    unsigned int *src_pointer_indices,
    unsigned int *dst_indices,
    unsigned int *level,
    unsigned int current_level,
    unsigned int *new_vertex_visited
) {
    unsigned int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex < VERTEX_LENGTH && level[vertex] == current_level - 1) {
        for (unsigned int edge = src_pointer_indices[vertex];
            edge < src_pointer_indices[vertex + 1]; ++edge) {
                unsigned int neighbor = dst_indices[edge];
                // Ignore if already visited.
                if (level[neighbor] != UINT_MAX)
                    continue;
                level[neighbor] = current_level;
                new_vertex_visited[0] = 1;
        }
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
    unsigned int *csr_src_pointer_indices_d, *csr_dst_indices_d, *level_d, *new_vertex_visited_d;

    size_t size_src_pointer_indices = (VERTEX_LENGTH + 1) * sizeof(unsigned int);
    size_t size_dst_indices = edge_length * sizeof(unsigned int);
    size_t size_vertex = VERTEX_LENGTH * sizeof(unsigned int);

    cudaMalloc((void**)&csr_src_pointer_indices_d, size_src_pointer_indices);
    cudaMemcpy(csr_src_pointer_indices_d, csr_src_pointer_indices_h, size_src_pointer_indices, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&csr_dst_indices_d, size_dst_indices);
    cudaMemcpy(csr_dst_indices_d, csr_dst_indices_h, size_dst_indices, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&level_d, size_vertex);
    // Initialize with "not visited" value.
    cudaMemset(level_d, UINT_MAX, size_vertex);

    // new_vertex_visited is used to determine whether we need to do another iteration.
    cudaMalloc((void**)&new_vertex_visited_d, sizeof(unsigned int));

    // Invoke kernel per iteration.
    dim3 dimBlock(THREADS_NUM_PER_BLOCK);
    dim3 dimGrid(BLOCK_NUM);

    unsigned int *new_vertex_visited_h = (unsigned int*)malloc(sizeof(unsigned int));
    // Initialize with 1.
    new_vertex_visited_h[0] = 1;

    // Assign level[root] to 0 (i.e., visited).
    cudaMemset(level_d + root, 0, sizeof(unsigned int));
    unsigned int current_level = 1;

    while (new_vertex_visited_h[0] != 0) {
        cudaMemset(new_vertex_visited_d, 0, sizeof(unsigned int));

        BfsTopDownKernel<<<dimGrid, dimBlock>>>(
            csr_src_pointer_indices_d,
            csr_dst_indices_d,
            level_d,
            current_level,
            new_vertex_visited_d
        );

        cudaMemcpy(new_vertex_visited_h, new_vertex_visited_d, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        ++current_level;
    }

    free(new_vertex_visited_h);

    // Copy level from the device memory.
    cudaMemcpy(level_h, level_d, size_vertex, cudaMemcpyDeviceToHost);

    // Free device arrays.
    cudaFree(csr_src_pointer_indices_d);
    cudaFree(csr_dst_indices_d);
    cudaFree(level_d);
    cudaFree(new_vertex_visited_d);
}


int main() {
    // Generate the CSR graph; all values are 1s.
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
        printf("%d ", level[i]);
    printf("\n");

    return 0;
}