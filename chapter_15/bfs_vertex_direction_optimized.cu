/**
 *  Perform direction-optimized vertex-centric BFS, corresponds to chapter 15.3 & exercise 2. 
 */
#include <stdio.h>
#include <limits.h>
#include <cuda_runtime.h>

#define VERTEX_LENGTH 9
#define THREADS_NUM_PER_BLOCK 4
#define BLOCK_NUM ceil(VERTEX_LENGTH * 1.0 / THREADS_NUM_PER_BLOCK)


struct CSRGraph {
    unsigned int *src_pointer_indices;
    unsigned int *dst_indices;
};

struct CSCGraph {
    unsigned int *dst_pointer_indices;
    unsigned int *src_indices;
};


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


__global__
void BfsBottomUpKernel(
    unsigned int *dst_pointer_indices,
    unsigned int *src_indices,
    unsigned int *level,
    unsigned int current_level,
    unsigned int *new_vertex_visited
) {
    unsigned int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex < VERTEX_LENGTH && level[vertex] == UINT_MAX) {
        for (unsigned int edge = dst_pointer_indices[vertex];
            edge < dst_pointer_indices[vertex + 1]; ++edge) {
                unsigned int neighbor = src_indices[edge];
                if (level[neighbor] == current_level - 1) {
                    level[vertex] = current_level;
                    new_vertex_visited[0] = 1;
                    break;
                }
        }
    }
}


/**
 * Perform a direction-optimized BFS. 
 * 
 * It switches from push to pull implementation when it hits the level defined as level_to_switch_direction.
 * For example, if level_to_switch_direction = 2, the function invokes push kernel for level 0 - 1 and switches to
 * pull kernel for the rest of the levels.
 * 
 */
void runDirectionOptimizedBFS(
    unsigned int *csr_src_pointer_indices_h,
    unsigned int *csr_dst_indices_h,
    unsigned int *csc_dst_pointer_indices_h,
    unsigned int *csc_src_indices_h,
    unsigned int edge_length,
    unsigned int *level_h,
    unsigned int root,
    unsigned int level_to_switch_direction
) {
    // Load and copy host variables to device memory.
    size_t size_pointer_indices = (VERTEX_LENGTH + 1) * sizeof(unsigned int);
    size_t size_indices = edge_length * sizeof(unsigned int);
    size_t size_vertex = VERTEX_LENGTH * sizeof(unsigned int);

    // Load CSR graph components.
    unsigned int *csr_src_pointer_indices_d, *csr_dst_indices_d, *level_d, *new_vertex_visited_d;

    cudaMalloc((void**)&csr_src_pointer_indices_d, size_pointer_indices);
    cudaMemcpy(csr_src_pointer_indices_d, csr_src_pointer_indices_h, size_pointer_indices, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&csr_dst_indices_d, size_indices);
    cudaMemcpy(csr_dst_indices_d, csr_dst_indices_h, size_indices, cudaMemcpyHostToDevice);

    // Load CSC graph components.
    unsigned int *csc_dst_pointer_indices_d, *csc_src_indices_d;
    cudaMalloc((void**)&csc_dst_pointer_indices_d, size_pointer_indices);
    cudaMemcpy(csc_dst_pointer_indices_d, csc_dst_pointer_indices_h, size_pointer_indices, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&csc_src_indices_d, size_indices);
    cudaMemcpy(csc_src_indices_d, csc_src_indices_h, size_indices, cudaMemcpyHostToDevice);

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

        if (current_level < level_to_switch_direction)
            BfsTopDownKernel<<<dimGrid, dimBlock>>>(
                csr_src_pointer_indices_d,
                csr_dst_indices_d,
                level_d,
                current_level,
                new_vertex_visited_d
            );
        else
            BfsBottomUpKernel<<<dimGrid, dimBlock>>>(
                csc_dst_pointer_indices_d,
                csc_src_indices_d,
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
    // Generate the CSR graph; all values (edge weight) are 1s.
    unsigned int src_pointer_indices[] = {0, 2, 4, 7, 9, 11, 12, 13, 15, 15};
    unsigned int dst_indices[] = {1, 2, 3, 4, 5, 6, 7, 4, 8, 5, 8, 6, 8, 0, 6};

    // Generate the CSC graph; all values (edge weight) are 1s.
    unsigned int dst_pointer_indices[] = {0, 1, 2, 3, 4, 6, 8, 11, 12, 15};
    unsigned int src_indices[] = {7, 0, 0, 1, 1, 3, 2, 4, 2, 5, 7, 2, 3, 4, 6};

    unsigned int edge_length = 15;

    unsigned int root = 2;
    unsigned int level_to_switch_direction = 2;
    unsigned int *level = (unsigned int*)malloc(VERTEX_LENGTH * sizeof(unsigned int));
    runDirectionOptimizedBFS(
        src_pointer_indices,
        dst_indices,
        dst_pointer_indices,
        src_indices,
        edge_length,
        level,
        root,
        level_to_switch_direction
    );

    for (unsigned int i = 0; i < VERTEX_LENGTH; ++i)
        printf("%u ", level[i]);
    printf("\n");

    return 0;
}