# Question 1

`kernel_functions.cu` contains two different matrix multiplication kernel functions:

- `Question1AKernel()`: each thread produces one row output
- `Question1BKernel()`: each thread produces one column output

Let `M`, `N` denote the first and second matrix to be multiplied, respectively, outputting a matrix `Out`.

To produce one row output in each thread, each thread in `Question1AKernel()` defines the row in matrix `M` and `Out`. It then calculates the dot product between the row in matrix `M` and for *each* column in matrix `N` to produce one row output. Meanwhile, each thread in `Question1BKernel()` defines the column in matrix `N` and `Out`. The function then calcultes the dot product between *each* row in matrix `M` and the column in matrix `N` to produce one column output.

Given that CUDA C has the row major layout characteristic,`Question1BKernel()` is able to utilize it during the dot product calculation, so we can expect that `Question1BKernel()` would be faster than `Question1AKernel()`.

## Running with smaller matrix width to check correctness.

File: `check_correctness.cu`

I prepared two identical matrices of size 100 x 100 with values `0..100*100 % 10` produced by the following code (I used Python just because I'm more familiar with it.)

```
import numpy as np

width = 100
m = np.arange(width*width).reshape((width, width)) % 10
n = np.arange(width*width).reshape((width, width)) % 10

# Get the matrix multiplication result.
output = m @ n
```

The input and output were then stored in `.txt` files:

- `chapter_3_matmul_input.txt` contains `m` or `n` (recall that they're identical)
- `chapter_3_matmul_output.txt` contains `output`

Each file contains 2 rows:

- row 1: one integer indicating the width of the matrix
- row 2: `width * width` number of elements of the matrix, each separated by a space. The row ends with an endline.

Running with Q1A and Q1B kernels prints total durations of `~0.0006s` and `~0.0003s`, respectively.

## Running with larger matrix width.

File: `run_question.cu`

The current width is set to `5000`. Running with Q1A and Q1B kernel prints total durations of `~14.18s` and `~6.5s`, respectively.

## Summary

Question 1B kernel consistently leads to faster run time (around 1/2 of question 1A kernel runtime) based on the tests above. One potential improvement would be to run multiple times and compute runtime average instead of relying only on one run.