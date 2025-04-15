# Question 1

`kernel_functions.cu` contains two different matrix multiplication kernel functions:

- `Question1AKernel()`: each thread produces one row output
- `Question1BKernel()`: each thread produces one column output

In terms of the pros and cons of each function, my first guess is that `Question1AKernel()` would be faster when handling larger matrices compared to `Question1BKernel()` given the CUDA C's row major layout characteristic. 

## Running with smaller matrix width to check correctness.

File: `check_correctness.cu`

I prepared two identical matrices of size 100 x 100 with values `0..100*100` produced by the following code (I used Python just because I'm more familiar with it.)

```
import numpy as np

width = 100
m = np.arange(width*width).reshape((width, width))
n = np.arange(width*width).reshape((width, width))

# Get the matrix multiplication result.
output = m @ n
```

The input and output were then stored in `.txt` files:

- `chapter_3_matmul_input.txt` contains `m` or `n` (recall that they're identical)
- `chapter_3_matmul_output.txt` contains `output`

Each file contains 2 rows:

- row 1: one integer indicating the width of the matrix
- row 2: `width * width` number of elements of the matrix, each separated by a space. The row ends with an endline.

Running with Q1A and Q1B kernels prints total durations of `~0.0008s` and `~0.0005s`, respectively.

## Running with larger matrix width.

File: `run_question.cu`

The current width is set to `5000`. Running with Q1A and Q1B kernel prints total durations of `~13.7s` and `~6.5s`, respectively.

## Summary(?)

Surprisingly, question 1B kernel consistently leads to faster run time (around 1/2 of question 1A kernel runtime)... My plan is to keep reading the book to see if I'm missing something or to eventually ask online...