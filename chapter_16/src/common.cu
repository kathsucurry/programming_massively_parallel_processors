#include <cuda_runtime.h>
#include <stdio.h> 


cudaError_t cudaMallocCheck(void **dev_ptr, size_t size) {
    cudaError_t cuda_malloc_return = cudaMalloc(dev_ptr, size);
    if (cuda_malloc_return == cudaErrorMemoryAllocation) {
        printf("CUDA malloc failed!\n");
        exit(EXIT_FAILURE);
    }
    return cuda_malloc_return;
}


void *mallocCheck(size_t size) {
    void *malloc_return = malloc(size);
    if (malloc_return == NULL) {
        printf("Malloc failed!\n");
        exit(EXIT_FAILURE);
    }
    return malloc_return;
}
