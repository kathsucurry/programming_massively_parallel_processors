#include <stdio.h>
#include <cuda_runtime.h>

#define gpu_check_error(ans) { gpu_assert((ans), __LINE__); }


void gpu_assert(cudaError_t code, int line);
