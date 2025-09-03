#include <stdio.h>
#include <cuda_runtime.h>

#define gpu_check_error(ans) { gpu_assert((ans), __LINE__); }
#define gpu_check_error_d(ans) { gpu_assert_d((ans), __LINE__); }


void gpu_assert(cudaError_t code, int line);
__device__ void gpu_assert_d(cudaError_t code, int line);
float2 *get_random_points(int count_points, int seed);