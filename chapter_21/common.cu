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


float2 *get_random_points(int count_points, int seed) {
   float2 *points = (float2 *)malloc(count_points * sizeof(float2));
   for (int i = 0; i < count_points; ++i) {
      points[i].x = (float)rand() / (float)RAND_MAX;
      points[i].y = (float)rand() / (float)RAND_MAX;
   }
   return points;
}
