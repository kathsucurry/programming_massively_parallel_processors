#include "kernel_functions.cu"


/**
 * Parse a line read from file into a matrix.
 */
 void parse_line_to_matrix(char * line, int * matrix, int matrix_width) {
    const int INIT_VALUE = -1;
    int matrix_index = 0, current_value = INIT_VALUE, char_index = 0;
    while (true) {
        if (line[char_index] == ' ' || line[char_index] == '\0' || line[char_index] == '\n') {
            if (current_value != INIT_VALUE) {
                matrix[matrix_index] = current_value;
                current_value = INIT_VALUE;
                ++matrix_index;
                ++char_index;
                continue;
            }

            if (line[char_index] == '\0' || line[char_index] == '\n') {
                break;
            }
        }

        int value = line[char_index] - '0';
        if (current_value == INIT_VALUE) {
            current_value = value;
        } else {
            current_value = current_value * 10 + value;
        }
        ++char_index;
    }
    assert (matrix_width * matrix_width == matrix_index);
}



/**
 * Reads a square matrix from file and returns the matrix width.
 *
 * The file contains 2 lines:
 * - Line 1: an integer indicating the width of the matrix.
 * - Line 2: the [width x width] element of matrix (all rows appended into one row) where each
 *           value is separated by a space. The line always ends with an end of line.
 *
 */
int read_matrix_from_file(int ** matrix, size_t max_buffer_size, const char* filename) {
    FILE* file_pointer = fopen(filename, "r");

    if (file_pointer == NULL) {
        fprintf(stderr, "Failed to open file with name %s\n", filename);
        return -1;
    }

    // Allocate memory for the line buffer.
    char *buffer = (char *) malloc(max_buffer_size);

    // Initialize the line counter;
    int line_counter = 0;
    int matrix_width = NULL;
    while (fgets(buffer, max_buffer_size, file_pointer)) {
        if (line_counter == 0) {
            sscanf(buffer, "%d\n", &matrix_width);
            * matrix = (int *) malloc(matrix_width * matrix_width * sizeof(int));
        } else if (line_counter == 1) {
            parse_line_to_matrix(buffer, * matrix, matrix_width);
        }
        ++line_counter;
    }
    free(buffer);
    fclose(file_pointer);

    return matrix_width;
}


void check_matmul_results(int * out_matrix, int * expected_out_matrix, int width) {
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            if (out_matrix[i * width + j] != expected_out_matrix[i * width + j]) {
                printf("Incorrect results.\n");
                return;
            }
        }
    }
    printf("Correct results.\n");
}


void run_kernel_and_check(
    int * matrix_M,
    int * matrix_N,
    int * matrix_expected_Out,
    int Width,
    MatrixMultiplicationFuction* matmul_kernel,
    const char * label
) {
    printf("Running %s...\n", label);

    int * matrix_Out = (int *) malloc(Width * Width * sizeof(int));
    
    runMatrixMultiplication(matrix_M, matrix_N, matrix_Out, Width, matmul_kernel);
    check_matmul_results(matrix_Out, matrix_expected_Out, Width);
    
    free(matrix_Out);
    return;
}


int main() {
    // Matrices are to be stored in row-major order.
    int * matrix_M;
    int * matrix_N;
    int * matrix_expected_Out;

    // Read inputs; we already know the approximate length of the characters.
    size_t input_max_length = 50000, output_max_length = 110000;

    int matrix_width_M = read_matrix_from_file(&matrix_M, input_max_length, "chapter_3_matmul_input.txt");
    int matrix_width_N = read_matrix_from_file(&matrix_N, input_max_length, "chapter_3_matmul_input.txt");
    int matrix_width_expected_Out = read_matrix_from_file(&matrix_expected_Out, output_max_length, "chapter_3_matmul_output.txt");
    
    assert (matrix_width_M == matrix_width_N && matrix_width_N == matrix_width_expected_Out);
    int Width = matrix_width_M;

    // Update the kernel function/label accordingly.
    run_kernel_and_check(
        matrix_M,
        matrix_N,
        matrix_expected_Out,
        Width,
        Question1AKernel,
        "Question 1A Kernel"
    );

    free(matrix_M);
    free(matrix_N);
    free(matrix_expected_Out);

    return 0;
}