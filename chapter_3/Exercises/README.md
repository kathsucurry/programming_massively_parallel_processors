# Question 1

`question_1.cu` contains two different matrix multiplication kernel functions:

- `Question1AKernel()`: each thread produces one row output
- `Question1BKernel()`: each thread produces one column output

In terms of the pros and cons of each function, my first guess is that `Question1AKernel()` would be faster when handling larger matrices compared to `Question1BKernel()` given the CUDA C's row major layout characteristic. 

To test it, I prepared two identical matrices of size 100 x 100 with values `0..100*100` produced by the following code (I used Python just because I'm more familiar with it.)

```
import numpy as np

size = 100
m = np.arange(size*size).reshape((size, size))
n = np.arange(size*size).reshape((size, size))

# Get the matrix multiplication result.
output = m @ n
```

The input and output were then stored in `.txt` files:

- `chapter_3_matmul_input.txt` contains `m` or `n` (recall that they're identical)
- `chapter_3_matmul_output.txt` contains `output`

where each file contains 1 row: the matrix linearized using the row-major layout, each value separated by a space, and the last value is followed by an endline.