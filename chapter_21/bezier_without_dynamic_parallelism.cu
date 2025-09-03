/**
 *  Perform Bezier curve calculation without dynamic parallelism, corresponds to Fig. 21.6.
 */
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include "common.cuh"


#define MAX_TESS_POINTS 32
#define COUNT_POINTS 3


struct BezierLine {
    float2 control_points[COUNT_POINTS]; // The 3 control points of the Bezier line.
    float2 vertex_positions[MAX_TESS_POINTS]; // The vertex position array to tessellate into.
    int count_vertices;
};


__device__ float get_length(float2 P) {
    return sqrtf(P.x * P.x + P.y * P.y);
}


/**
 * Taken from the CUDA sample code: https://github.com/NVIDIA/cuda-samples/blob/master/Samples/3_CUDA_Features/cdpBezierTessellation/BezierLineCDP.cu#L91.
 */
__device__ float get_curvature(float2 control_points[COUNT_POINTS]) {
    float2 numerator = {0, 0};
    numerator.x = control_points[1].x - 0.5f * (control_points[0].x + control_points[2].x);
    numerator.y = control_points[1].y - 0.5f * (control_points[0].y + control_points[2].y);

    float2 denominator = {0, 0};
    denominator.x = control_points[2].x - control_points[0].x;
    denominator.y = control_points[2].y - control_points[0].y;

    return get_length(numerator) / get_length(denominator);
}


__global__ void computeBezierLinesKernel(BezierLine *b_lines, int count_lines) {
    if (blockIdx.x < count_lines) {
        // Compute the curvature of the line.
        float curvature = get_curvature(b_lines[blockIdx.x].control_points);

        // Determine the number of vertices to tessellate into based on curvature.
        int count_tess_points = min(max((int)(curvature * 16.0f), 4), MAX_TESS_POINTS);
        b_lines[blockIdx.x].count_vertices = count_tess_points;

        // Loop through vertices to be tessellated.
        for (int tess_point_offset = 0 ; tess_point_offset < count_tess_points; tess_point_offset += blockDim.x) {
            int vertex_i = tess_point_offset + threadIdx.x;
            if (vertex_i < count_tess_points) {
                float t = (float)vertex_i / (float)(count_tess_points - 1);
                float one_minus_t = 1.0f - t;
                // Compute quadratic Bezier coefficients.
                float quad_bezier_coeffs[COUNT_POINTS];
                quad_bezier_coeffs[0] = one_minus_t * one_minus_t;
                quad_bezier_coeffs[1] = 2.0f * t * one_minus_t;
                quad_bezier_coeffs[2] = t * t;
                float2 position = {0, 0};
                // Add the contribution of the control points to the position.
                for (int control_point_i = 0; control_point_i < COUNT_POINTS; ++control_point_i) {
                    position.x += quad_bezier_coeffs[control_point_i] * b_lines[blockIdx.x].control_points[control_point_i].x;
                    position.y += quad_bezier_coeffs[control_point_i] * b_lines[blockIdx.x].control_points[control_point_i].y;
                }

                b_lines[blockIdx.x].vertex_positions[vertex_i] = position;
            }
        }
    }
}


void computeBezierLines  (
    BezierLine *b_lines,
    int count_lines
) {
    // Load and copy host variables to device memory.
    BezierLine *b_lines_d = (BezierLine *)malloc(count_lines * sizeof(BezierLine));
    gpu_check_error(cudaMalloc(&b_lines_d, count_lines * sizeof(BezierLine)));
    gpu_check_error(cudaMemcpy(b_lines_d, b_lines, count_lines * sizeof(BezierLine), cudaMemcpyHostToDevice));

    computeBezierLinesKernel<<<count_lines, MAX_TESS_POINTS>>>(b_lines_d, count_lines);

    gpu_check_error(cudaMemcpy(b_lines, b_lines_d, count_lines * sizeof(BezierLine), cudaMemcpyDeviceToHost));

    cudaFree(b_lines_d);
}


int main() {
    // Prepare sets of control points.
    int count_lines = 2;

    BezierLine *b_lines = (BezierLine *)malloc(count_lines * sizeof(BezierLine));
    for (int line_i = 0; line_i < count_lines; ++line_i) {
        float2 *points = get_random_points(3, line_i);
        for (int point_i = 0; point_i < COUNT_POINTS; ++point_i) {
            b_lines[line_i].control_points[point_i].x = points[point_i].x;
            b_lines[line_i].control_points[point_i].y = points[point_i].y;
        }
        free(points);
    }
    computeBezierLines(b_lines, count_lines);

    // Print results.
    for (int line_i = 0; line_i < count_lines; ++line_i) {
        printf("Line %d:\n", line_i);
        for (int vertex_i = 0; vertex_i < b_lines[line_i].count_vertices; ++vertex_i) {
            float2 vertex_position = b_lines[line_i].vertex_positions[vertex_i];
            printf(">>> Vertex %2d: (%5.2f, %5.2f)\n", vertex_i, vertex_position.x, vertex_position.y);
        }
    }

    free(b_lines);

    return 0;
}