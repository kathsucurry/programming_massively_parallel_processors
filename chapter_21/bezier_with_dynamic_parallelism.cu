/**
 *  Perform Bezier curve calculation with dynamic parallelism, corresponds to Fig. 21.7.
 */
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include "common.cuh"


#define MAX_TESS_POINTS 32
#define COUNT_THREADS 32
#define COUNT_POINTS 3


struct BezierLine {
    float2 control_points[COUNT_POINTS]; // The 3 control points of the Bezier line.
    float2 *vertex_positions; // The vertex position array to tessellate into.
    int count_vertices;
};


struct VertexPositions {
    float2 *positions;
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


/**
 * Child kernel: performs the calculation.
 */
__global__ void computeBezierLinesChildKernel(int line_idx, BezierLine *b_lines, int count_tess_points) {
    int vertex_i = threadIdx.x + blockDim.x * blockIdx.x;
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
            position.x += quad_bezier_coeffs[control_point_i] * b_lines[line_idx].control_points[control_point_i].x;
            position.y += quad_bezier_coeffs[control_point_i] * b_lines[line_idx].control_points[control_point_i].y;
        }
        b_lines[line_idx].vertex_positions[vertex_i] = position;
    }
}


/**
 * Parent kernel: discovers the amount of work to be done for each control point.
 */
__global__ void computeBezierLinesParentKernel(BezierLine *b_lines, int count_lines) {
    int line_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (line_idx < count_lines) {
        // Compute the curvature of the line.
        float curvature = get_curvature(b_lines[line_idx].control_points);

        // Determine the number of vertices to tessellate into based on curvature.
        int count_tess_points = min(max((int)(curvature * 16.0f), 4), MAX_TESS_POINTS);
        gpu_check_error_d(cudaMalloc((void **)&b_lines[line_idx].vertex_positions, count_tess_points * sizeof(float2)));
        b_lines[line_idx].count_vertices = count_tess_points;

        // Call the child kernel to compute the tessellated points for each line.
        computeBezierLinesChildKernel<<<ceil((float)b_lines[line_idx].count_vertices / (float)MAX_TESS_POINTS), MAX_TESS_POINTS>>>(
            line_idx, b_lines, b_lines[line_idx].count_vertices);
    }
}


__global__ void freeVertexMemKernel(BezierLine *b_lines, int count_lines) {
    int line_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (line_idx < count_lines)
        gpu_check_error_d(cudaFree(b_lines[line_idx].vertex_positions));
}


__global__ void copyVerticePositionsKernel(VertexPositions *positions, BezierLine *b_lines, int count_lines) {
    int line_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (line_idx < count_lines)
        memcpy(positions[line_idx].positions, b_lines[line_idx].vertex_positions, b_lines[line_idx].count_vertices * sizeof(float2));
}


void computeBezierLines  (
    BezierLine *b_lines,
    int count_lines
) {
    // Load and copy host variables to device memory.
    BezierLine *b_lines_d, *b_lines_updated;
    gpu_check_error(cudaMalloc((void **)&b_lines_d, count_lines * sizeof(BezierLine)));
    gpu_check_error(cudaMemcpy(b_lines_d, b_lines, count_lines * sizeof(BezierLine), cudaMemcpyHostToDevice));

    computeBezierLinesParentKernel<<<ceil(count_lines * 1.0 / COUNT_THREADS), COUNT_THREADS>>>(b_lines_d, count_lines);
    gpu_check_error(cudaDeviceSynchronize());

    b_lines_updated = (BezierLine *)malloc(count_lines * sizeof(BezierLine));
    gpu_check_error(cudaMemcpy(b_lines_updated, b_lines_d, count_lines * sizeof(BezierLine), cudaMemcpyDeviceToHost));
    
    VertexPositions *positions_d;
    cudaMalloc((void**)&positions_d, count_lines * sizeof(VertexPositions));

    float2 *positions_values_d[count_lines];
    for (int line_i = 0; line_i < count_lines; ++line_i) {
        b_lines[line_i].count_vertices = b_lines_updated[line_i].count_vertices;
        b_lines[line_i].vertex_positions = (float2 *)malloc(b_lines[line_i].count_vertices * sizeof(float2));
        gpu_check_error(cudaMalloc((void **)&(positions_values_d[line_i]), b_lines[line_i].count_vertices * sizeof(float2)));
        gpu_check_error(cudaMemcpy(&(positions_d[line_i].positions), &(positions_values_d[line_i]), sizeof(float2 *), cudaMemcpyHostToDevice));
    }
    free(b_lines_updated);
    
    // Dynamically allocated memory in kernel cannot be directly transferred to host memory, so we'd need to copy the data from
    // the region allocated by the in-kernel malloc to another region allocated by the host-based API.
    copyVerticePositionsKernel<<<ceil(count_lines * 1.0 / COUNT_THREADS), COUNT_THREADS>>>(positions_d, b_lines_d, count_lines);
    gpu_check_error(cudaDeviceSynchronize());
    
    for (int line_i = 0; line_i < count_lines; ++line_i) {
        gpu_check_error(
            cudaMemcpy(b_lines[line_i].vertex_positions, positions_values_d[line_i], b_lines[line_i].count_vertices * sizeof(float2), cudaMemcpyDeviceToHost)
        );
        gpu_check_error(cudaFree(positions_values_d[line_i]));
    }
    gpu_check_error(cudaFree(positions_d));

    freeVertexMemKernel<<<ceil(count_lines * 1.0 / COUNT_THREADS), COUNT_THREADS>>>(b_lines_d, count_lines);

    gpu_check_error(cudaFree(b_lines_d));
}


int main() {
    // Prepare sets of control points.
    float control_points_set_1[] = {0.0f, 3.0f, 0.0f, 8.0f, 5.0f, 8.0f};
    float control_points_set_2[] = {1.0f, 8.0f, 3.0f, 13.0f, 5.0f, 7.0f};
    int count_lines = 2;

    BezierLine *b_lines = (BezierLine *)malloc(count_lines * sizeof(BezierLine));
    for (int point_i = 0; point_i < COUNT_POINTS * 2; ++point_i) {
        if (point_i % 2 == 0) {
            b_lines[0].control_points[point_i / 2].x = control_points_set_1[point_i];
            b_lines[1].control_points[point_i / 2].x = control_points_set_2[point_i];
        } else {
            b_lines[0].control_points[point_i / 2].y = control_points_set_1[point_i];
            b_lines[1].control_points[point_i / 2].y = control_points_set_2[point_i];
        }
    }
    b_lines[0].vertex_positions = nullptr;
    b_lines[1].vertex_positions = nullptr;

    computeBezierLines(b_lines, count_lines);

    // Print results and free memory.
    for (int line_i = 0; line_i < count_lines; ++line_i) {
        printf("Line %d: %d vertices\n", line_i, b_lines[line_i].count_vertices);
        for (int vertex_i = 0; vertex_i < b_lines[line_i].count_vertices; ++vertex_i) {
            float2 vertex_position = b_lines[line_i].vertex_positions[vertex_i];
            printf(">>> Vertex %2d: (%5.2f, %5.2f)\n", vertex_i, vertex_position.x, vertex_position.y);
        }
        free(b_lines[line_i].vertex_positions);
    }
    free(b_lines);

    return 0;
}