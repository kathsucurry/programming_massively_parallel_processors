#include "kernel_functions.cu"


int main() {
    // Matrices are to be stored in row-major order.
    int Width = 5000;
    int * matrix_M = (int *) malloc(Width * Width * sizeof(int));
    int * matrix_N = (int *) malloc(Width * Width * sizeof(int));

    for (int i = 0; i < Width * Width; ++i) {
        matrix_M[i] = i;
        matrix_N[i] = i;
    }

    // Update the kernel function/label accordingly.
    // The first run is only for warming up (i.e., initializing resources etc).
    run_kernel(
        matrix_M,
        matrix_N,
        Width,
        Question1AKernel,
        "Question 1A Kernel"
    );

    // This is where we're interested in the runtime.
    run_kernel(
        matrix_M,
        matrix_N,
        Width,
        Question1AKernel,
        "Question 1A Kernel"
    );

    free(matrix_M);
    free(matrix_N);
    return 0;
}