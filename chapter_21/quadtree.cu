/**
 *  Perform quadtree, corresponds to to chapter 21.4.
 */
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include "common.cuh"


#define MAX_DEPTH 3
#define MIN_POINTS_IN_NODE 2
#define COUNT_THREADS_PER_BLOCK 256


class Points {
    float *x_array;
    float *y_array;

    public:
    // Constructor.
    __host__ __device__ Points() : x_array(NULL), y_array(NULL) {}

    // Constructor.
    __host__ __device__ Points(float *x, float *y) : x_array(x), y_array(y) {}

    // Get a point.
    __host__ __device__ __forceinline__ float2 get_point(int index) const {
        return make_float2(x_array[index], y_array[index]);
    }

    // Set a point.
    __host__ __device__ __forceinline__ void set_point(int index, const float2 &p) {
        x_array[index] = p.x;
        y_array[index] = p.y;
    }

    // Set the pointers.
    __host__ __device__ __forceinline__ void set(float *x, float *y) {
        x_array = x;
        y_array = y;
    }
};


class BoundingBox {
    // Extreme points of the bounding box.
    float2 point_min;
    float2 point_max;

    public:
    // Constructor: create a unit box.
    __host__ __device__ BoundingBox() {
        point_min = make_float2(0.0f, 0.0f);
        point_max = make_float2(1.0f, 1.0f);
    }

    // Compute the center of the bounding box.
    __host__ __device__ void compute_center(float2 &center) const {
        center.x = 0.5f * (point_min.x + point_max.x);
        center.y = 0.5f * (point_min.y + point_max.y);
    }

    // The points of the box.
    __host__ __device__ __forceinline__ const float2  &get_max() const {
        return point_max;
    }

    __host__ __device__ __forceinline__ const float2 &get_min() const {
        return point_min;
    }

    // Check if a box contains a point.
    __host__ __device__ bool contains(const float2 &p) const {
        return p.x >= point_min.x && p.x < point_max.x && p.y >= point_min.y && p.y < point_max.y;
    }

    // Define the bounding box.
    __host__ __device__ void set(float min_x, float min_y, float max_x, float max_y) {
        point_min.x = min_x;
        point_min.y = min_y;
        point_max.x = max_x;
        point_max.y = max_y;
    }
};


class QuadtreeNode {
    // The identifier of the node.
    int node_id;
    // The bounding box of the tree.
    BoundingBox bbox;
    // The range of points.
    int point_range_begin, point_range_end;

    bool is_used;

    public:
    // Constructor.
    __host__ __device__ QuadtreeNode() : node_id(0), point_range_begin(0), point_range_end(0), is_used(false) {}

    // The id of a node at its level.
    __host__ __device__ int id() const {
        return node_id;
    }

    __host__ __device__ void set_id(int new_id) {
        is_used = true;
        node_id = new_id;
    }

    // The bounding box.
    __host__ __device__ __forceinline__ const BoundingBox &bounding_box() const {
        return bbox;
    }

    __host__ __device__ __forceinline__ void set_bounding_box(float min_x, float min_y, float max_x, float max_y) {
        bbox.set(min_x, min_y, max_x, max_y);
    }

    // The number of points in the tree.
    __host__ __device__ __forceinline__ int num_points() const {
        return point_range_end - point_range_begin;
    }

    // The range of points in the tree.
    __host__ __device__ __forceinline__ int points_begin() const {
        return point_range_begin;
    }

    __host__ __device__ __forceinline__ int points_end() const {
        return point_range_end;
    }

    __host__ __device__ __forceinline__ void set_range(int begin, int end) {
        point_range_begin = begin;
        point_range_end   = end;
    }

    __host__ __device__ __forceinline__ bool used() const {
        return is_used;
    }
};

// Algorithm parameters.
struct Parameters {
    // Choose the right set of points to use as in/out.
    int point_selector;
    // The number of nodes at a given level (4^k for level k);
    int num_nodes_at_this_level;
    // The recursion depth.
    int depth;
    // The max value for depth.
    const int max_depth;
    // The min number of points in a node to stop recursion.
    const int min_points_per_node;

    // Constructor set to default value.
    __host__ __device__ Parameters(int max_depth, int min_points_per_node) :
        point_selector(0),
        num_nodes_at_this_level(1),
        depth(0),
        max_depth(max_depth),
        min_points_per_node(min_points_per_node) {}

    // Copy constructor; changes the values for next iteration.
    __host__ __device__ Parameters(const Parameters &params, bool) :
        point_selector((params.point_selector + 1) % 2),
        num_nodes_at_this_level(4 * params.num_nodes_at_this_level),
        depth(params.depth + 1),
        max_depth(params.max_depth),
        min_points_per_node(params.min_points_per_node) {}
};


__device__ bool check_num_points_and_depth(QuadtreeNode &node, Points *points, int num_points, Parameters params) {
    if (params.depth > params.max_depth || num_points <= params.min_points_per_node) {
        // Make sure points[0] contains all the points.
        if (params.point_selector == 1) {
            int it = node.points_begin(), end = node.points_end();
            for (it += threadIdx.x; it < end; it += blockDim.x)
                points[0].set_point(it, points[1].get_point(it));

        }
        return true;
    }
    return false;
}


// Count the number of points in each quadrant.
__device__ void count_points_in_children(const Points &in_points, int *smem, int range_begin, int range_end, float2 center) {
    // Initialize shared memory.
    if (threadIdx.x < 4) smem[threadIdx.x] = 0;
    __syncthreads();

    // Compute the number of points.
    for (int iter = range_begin + threadIdx.x; iter < range_end; iter += blockDim.x) {
        float2 p = in_points.get_point(iter);
        if (p.x < center.x && p.y >= center.y)
            atomicAdd(&smem[0], 1); // Top left points.
        if (p.x >= center.x && p.y >= center.y)
            atomicAdd(&smem[1], 1); // Top right points.
        if (p.x < center.x && p.y < center.y)
            atomicAdd(&smem[2], 1); // Bottom left points.
        if (p.x >= center.x && p.y < center.y)
            atomicAdd(&smem[3], 1); // Bottom right points.
    }
    __syncthreads();
}


// Scan quadrants' results to obtain reordering offset.
__device__ void scan_for_offsets(int node_points_begin, int *smem) {
    int *smem2 = &smem[4];
    if (threadIdx.x == 0) {
        for (int i = 0; i < 4; ++i)
            smem2[i] = i == 0 ? 0 : smem2[i-1] + smem[i-1]; // Sequential scan.
        for (int i = 0; i < 4; ++i)
            smem2[i] += node_points_begin; // Global offset.
    }
    __syncthreads();
}


__device__ void reorder_points(Points& out_points, const Points &in_points, int* smem, int range_begin, int range_end, float2 center) {
    int *smem2 = &smem[4];
    for (int iter = range_begin + threadIdx.x; iter < range_end; iter += blockDim.x) {
        int dest;
        float2 p = in_points.get_point(iter);
        if (p.x < center.x && p.y >= center.y)
            dest = atomicAdd(&smem2[0], 1);
        if (p.x >= center.x && p.y < center.y)
            dest = atomicAdd(&smem2[1], 1);
        if (p.x < center.x && p.y < center.y)
            dest = atomicAdd(&smem2[2], 1);
        if (p.x >= center.x && p.y < center.y)
            dest = atomicAdd(&smem2[3], 1);
        
        // Move point.
        out_points.set_point(dest, p);
    }
    __syncthreads();
}


__device__ void prepare_children(QuadtreeNode *children, QuadtreeNode &node, const BoundingBox &bbox, int *smem) {
    int child_offset= 4 * node.id(); // The offsets of the children at their level.

    // Set ids.
    children[child_offset + 0].set_id(child_offset + 0);
    children[child_offset + 1].set_id(child_offset + 1);
    children[child_offset + 2].set_id(child_offset + 2);
    children[child_offset + 3].set_id(child_offset + 3);

    // Points of the bounding box.
    const float2 &p_min = bbox.get_min();
    const float2 &p_max = bbox.get_max();
    float2 center;
    bbox.compute_center(center);

    // Set the bounding box of the children.
    children[child_offset + 0].set_bounding_box(p_min.x, center.y, center.x, p_max.y); // Top left.
    children[child_offset + 1].set_bounding_box(center.x, center.y, p_max.x, p_max.y); // Top right.
    children[child_offset + 2].set_bounding_box(p_min.x, p_min.y, center.x, center.y); // Bottom left.
    children[child_offset + 3].set_bounding_box(center.x, p_min.y, p_max.x, center.y); // bottom right.

    // Set the ranges of the children.
    children[child_offset + 0].set_range(node.points_begin(), smem[4 + 0]);
    children[child_offset + 1].set_range(smem[4 + 0], smem[4 + 1]);
    children[child_offset + 2].set_range(smem[4 + 1], smem[4 + 2]);
    children[child_offset + 3].set_range(smem[4 + 2], smem[4 + 3]);
}


__global__ void buildQuadtreeKernel(QuadtreeNode *nodes, Points *points, Parameters params) {
    // The first 4 store the number of points in each quadrant; the next 4 stores the quadrant offsets.
    __shared__ int smem[8];

    // The current node in the quadtree.
    QuadtreeNode &node = nodes[blockIdx.x];
    // if (threadIdx.x == 0)
    //     printf("%d %d\n", node.id(), blockIdx.x);
    int num_points = node.num_points();

    // Check the number of points and its depth.
    bool exit = check_num_points_and_depth(node, points, num_points, params);
    if (exit) return;

    // Compute the center of the bounding box of the points.
    const BoundingBox &bbox = node.bounding_box();
    float2 center;
    bbox.compute_center(center);

    // Range of points.
    int range_begin = node.points_begin();
    int range_end   = node.points_end();
    const Points &in_points = points[params.point_selector]; // Input points.
    Points &out_points      = points[(params.point_selector + 1) % 2]; // Output points.

    // Count the number of points in each child.
    count_points_in_children(in_points, smem, range_begin, range_end, center);

    // Scan the quadrants' results to know the reordering offset.
    scan_for_offsets(node.points_begin(), smem);

    // Move points.
    reorder_points(out_points, in_points, smem, range_begin, range_end, center);

    // Launch new blocks.
    if (threadIdx.x == blockDim.x - 1) {
        QuadtreeNode *children = &nodes[params.num_nodes_at_this_level];

        // Prepare the chidlren launch.
        prepare_children(children, node, bbox, smem);

        // Launch 4 children.
        buildQuadtreeKernel<<<4, blockDim.x, 8 * sizeof(int)>>>(children, points, Parameters(params, true));
    }
}


__global__ void prepareTwoBufferPointsKernel(Points *point, float *x1, float *y1, float *x2, float *y2) {
    point[0].set(x1, y1);
    point[1].set(x2, y2);
}


__global__ void getPointsFromTwoBufferPointsKernel(Points *points, float *x, float *y, int count_points) {
    int global_index = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = global_index; i < min(count_points, blockDim.x * blockIdx.x); i += blockDim.x) {
        float2 point = points[0].get_point(i);
        x[i] = point.x;
        y[i] = point.y;
    }
}


int main() {
    /* PREPARE POINTS */
    int count_points = 40;
    float2 *points = get_random_points(count_points, 0);
    float *points_x1 = (float *)malloc(count_points * sizeof(float));
    float *points_x2 = (float *)malloc(count_points * sizeof(float));
    float *points_y1 = (float *)malloc(count_points * sizeof(float));
    float *points_y2 = (float *)malloc(count_points * sizeof(float));
    for (int i = 0; i < count_points; ++i) {
        points_x1[i] = points[i].x;
        points_y1[i] = points[i].y;
        points_x2[i] = points[i].x;
        points_y2[i] = points[i].y;
    }
    free(points);

    Points *two_buffer_points_h = (Points *)malloc(2 * sizeof(Points));
    two_buffer_points_h[0].set(points_x1, points_y1);
    two_buffer_points_h[1].set(points_x2, points_y2);

    Points *two_buffer_points_d;
    float *points_x1_d, *points_y1_d, *points_x2_d, *points_y2_d;
    gpu_check_error(cudaMalloc((void **)&two_buffer_points_d, 2 * sizeof(Points)));
    gpu_check_error(cudaMemcpy(two_buffer_points_d, points, 2 * sizeof(Points), cudaMemcpyHostToDevice));

    // Load the x and y arrays into device.
    gpu_check_error(cudaMalloc((void **)&points_x1_d, count_points * sizeof(float)));
    gpu_check_error(cudaMemcpy(points_x1_d, points_x1, count_points * sizeof(float), cudaMemcpyHostToDevice));
    gpu_check_error(cudaMalloc((void **)&points_y1_d, count_points * sizeof(float)));
    gpu_check_error(cudaMemcpy(points_y1_d, points_y1, count_points * sizeof(float), cudaMemcpyHostToDevice));
    gpu_check_error(cudaMalloc((void **)&points_x2_d, count_points * sizeof(float)));
    gpu_check_error(cudaMemcpy(points_x2_d, points_x2, count_points * sizeof(float), cudaMemcpyHostToDevice));
    gpu_check_error(cudaMalloc((void **)&points_y2_d, count_points * sizeof(float)));
    gpu_check_error(cudaMemcpy(points_y2_d, points_y2, count_points * sizeof(float), cudaMemcpyHostToDevice));

    // Assign the x and y arrays to points.
    prepareTwoBufferPointsKernel<<<1, 1>>>(two_buffer_points_d, points_x1_d, points_y1_d, points_x2_d, points_y2_d);
    gpu_check_error(cudaGetLastError());
    gpu_check_error(cudaDeviceSynchronize());

    /* PREPARE QUADTREE NODES */
    int max_node_count = 0;
    int max_num_nodes_in_depth = 1;
    for (int i = 0; i <= MAX_DEPTH; ++i) {
        max_node_count += max_num_nodes_in_depth;
        max_num_nodes_in_depth *= 4;
    }

    // Prepare the root of the quadtree.
    QuadtreeNode root;
    root.set_range(0, count_points);
    root.set_id(0);
    // Recall that the bounding box is already set to [0, 0] - [1, 1];
    QuadtreeNode *nodes_d;
    gpu_check_error(cudaMalloc((void **)&nodes_d, max_node_count * sizeof(QuadtreeNode)));
    gpu_check_error(cudaMemcpy(nodes_d, &root, sizeof(QuadtreeNode), cudaMemcpyHostToDevice));

    /* PREPARE PARAMETERS */
    Parameters param(MAX_DEPTH, MIN_POINTS_IN_NODE);

    /* INVOKE KERNEL */
    buildQuadtreeKernel<<<1, COUNT_THREADS_PER_BLOCK>>>(nodes_d, two_buffer_points_d, param);
    gpu_check_error(cudaGetLastError());
    gpu_check_error(cudaDeviceSynchronize());

    /* LOAD RESULTS */
    // Points.
    getPointsFromTwoBufferPointsKernel<<<ceil(count_points * 1.0 / COUNT_THREADS_PER_BLOCK), COUNT_THREADS_PER_BLOCK>>>(
        two_buffer_points_d, points_x1_d, points_y1_d, count_points);
    gpu_check_error(cudaGetLastError());
    gpu_check_error(cudaDeviceSynchronize());
    gpu_check_error(cudaMemcpy(points_x1, points_x1_d, count_points * sizeof(float), cudaMemcpyDeviceToHost));
    gpu_check_error(cudaMemcpy(points_y1, points_y1_d, count_points * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Nodes.
    QuadtreeNode *nodes_h = (QuadtreeNode *)malloc(max_node_count * sizeof(QuadtreeNode));
    gpu_check_error(cudaMemcpy(nodes_h, nodes_d, max_node_count * sizeof(QuadtreeNode), cudaMemcpyDeviceToHost));
    gpu_check_error(cudaFree(nodes_d));

    for (int node_i = 0; node_i < max_node_count; ++node_i) {
        QuadtreeNode node = nodes_h[node_i];
        printf("Node index %2d with id %2d | range %2d : %2d | used: %d\n", node_i, node.id(), node.points_begin(), node.points_end(), node.used());
    }

    return 0;
}



