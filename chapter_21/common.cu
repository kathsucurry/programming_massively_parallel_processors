#include <stdio.h>
#include <cuda_runtime.h>

#include "common.cuh"


void gpu_assert(cudaError_t code, int line) {
   if (code != cudaSuccess) {
      printf("GPUassert: %s; line %d\n", cudaGetErrorString(code), line);
      exit(code);
   }
}