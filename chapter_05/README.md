# Chapter 5

This directory contains the following files:

- `matrix_multiplication_shared_mem.cu`: matrix multiplication from chapter 3 that utilizes shared memory
- `matrix_multiplication_rect.cu`: an extended version of `matrix_multiplication_shared_mem.cu` that allows performing matrix multiplication on matrices of any sizes (as long as they are valid to perform the process)
- `matrix_multiplication_dynamic_shared_size.cu`: an extended version of `matrix_multiplication_shared_mem.cu` that allows passing the shared memory size to the kernel function