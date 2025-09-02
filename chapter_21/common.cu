#include <stdio.h>
#include <cuda_runtime.h>

#include "common.cuh"


void gpu_assert(cudaError_t code, int line) {
   if (code != cudaSuccess) {
      printf("GPUassert: %u %s; line %d\n", code, cudaGetErrorString(code), line);
      exit(code);
   }
}


/**
 * Note that this function does not halt the execution.
 */
__device__ void gpu_assert_d(cudaError_t code, int line) {
   if (code != cudaSuccess) {
      printf("GPUassert: %u %s; line %d\n", code, cudaGetErrorString(code), line);
      return;
   }
}