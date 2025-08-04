/**
 *  Perform BFS top down, corresponds to chapter 15.3 (Fig. 15.6).
 */
#include <stdio.h>
#include <limits.h>
#include <cuda_runtime.h>

#define VERTEX_LENGTH 9
#define EDGE_LENGTH 15
#define THREADS_NUM_PER_BLOCK 4
#define BLOCK_NUM ceil(EDGE_LENGTH * 1.0 / THREADS_NUM_PER_BLOCK)


__global__
void BfsEdgeCentricKernel(
    unsigned int *src_indices,
    unsigned int *dst_indices,
    unsigned int *level,
    unsigned int current_level,
    unsigned int *new_vertex_visited
) {
    unsigned int edge = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge < EDGE_LENGTH) {
        unsigned int vertex_src = src_indices[edge];
        if (level[vertex_src] == current_level - 1) {
            unsigned int vertex_dst = dst_indices[edge];
            if (level[vertex_dst] == UINT_MAX) {
                level[vertex_dst] = current_level;
                *new_vertex_visited = 1;
            }
        }
    }
}


void runBFS(
    unsigned int *coo_src_indices_h,
    unsigned int *coo_dst_indices_h,
    unsigned int *level_h,
    unsigned int root
) {
    // Load and copy host variables to device memory.
    unsigned int *coo_src_indices_d, *coo_dst_indices_d, *level_d, *new_vertex_visited_d;

    size_t size_indices = EDGE_LENGTH * sizeof(unsigned int);
    size_t size_vertex = VERTEX_LENGTH * sizeof(unsigned int);

    cudaMalloc((void**)&coo_src_indices_d, size_indices);
    cudaMemcpy(coo_src_indices_d, coo_src_indices_h, size_indices, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&coo_dst_indices_d, size_indices);
    cudaMemcpy(coo_dst_indices_d, coo_dst_indices_h, size_indices, cudaMemcpyHostToDevice);

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

        BfsEdgeCentricKernel<<<dimGrid, dimBlock>>>(
            coo_src_indices_d,
            coo_dst_indices_d,
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
    cudaFree(coo_src_indices_d);
    cudaFree(coo_dst_indices_d);
    cudaFree(level_d);
    cudaFree(new_vertex_visited_d);
}


int main() {
    // Generate the COO graph; all values (edge weight) are 1s.
    unsigned int src_indices[] = {0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 6, 7, 7};
    unsigned int dst_indices[] = {1, 2, 3, 4, 5, 6, 7, 4, 8, 5, 8, 6, 8, 0, 6};

    unsigned int root = 2;
    unsigned int *level = (unsigned int*)malloc(VERTEX_LENGTH * sizeof(unsigned int));
    runBFS(
        src_indices,
        dst_indices,
        level,
        root
    );

    for (unsigned int i = 0; i < VERTEX_LENGTH; ++i)
        printf("%u ", level[i]);
    printf("\n");

    return 0;
}