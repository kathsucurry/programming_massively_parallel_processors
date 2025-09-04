#include "kernel_functions.cu"


int main() {
    // Matrices are to be stored in row-major order.
    int Width = 5000;
    float * matrix_M = (float *) malloc(Width * Width * sizeof(float));
    float * matrix_N = (float *) malloc(Width * Width * sizeof(float));

    for (int i = 0; i < Width * Width; ++i) {
        // Use a modulo to prevent overflow.
        matrix_M[i] = i % 10;
        matrix_N[i] = i % 10;
    }

    // Update the kernel function/label accordingly.
    // The first run is only for warming up (i.e., initializing resources etc).
    run_kernel(
        matrix_M,
        matrix_N,
        Width,
        Question1BKernel,
        "Question 1B Kernel"
    );

    // This is where we're interested in the runtime.
    run_kernel(
        matrix_M,
        matrix_N,
        Width,
        Question1BKernel,
        "Question 1B Kernel"
    );

    free(matrix_M);
    free(matrix_N);
    return 0;
}