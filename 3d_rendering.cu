#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstring>
#include "3d_rendering.cuh"
#include "cuda_spatial_grid.cuh"
#include "cuda_frustum_culling.cuh"

const int BLOCK_SIZE = 256;

// error checking macro - ensures immediate detection of cuda failures
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("cuda error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); } }

// helper functions for safe osmid access - both on host and device
__host__ __device__ inline uint64_t get_osmid_value(const OSMID& id) {
    // extract raw uint64 value from osmid class using reinterpret cast
    // osmid must have standard layout for this to work correctly
    return *reinterpret_cast<const uint64_t*>(&id);
}
__host__ __device__ inline bool osmid_less(const OSMID& a, const OSMID& b) {
    // comparison operator for osmid values - used in sorting algorithms
    return get_osmid_value(a) < get_osmid_value(b);
}
__host__ __device__ inline bool osmid_greater(const OSMID& a, const OSMID& b) {
    // reverse comparison for descending sort operations
    return get_osmid_value(a) > get_osmid_value(b);
}

// wrapper for building frustum culling with spatial grid
extern "C" void cuda_perform_frustum_culling_wrapper(gpu_building_data* d_buildings, int building_count, float4* d_planes, const cuda_grid& grid, int* d_visible_indices, int* d_visible_count) {
    // direct call to core frustum culling function
    // wrapper pattern allows for future expansion of functionality
    cuda_perform_frustum_culling(d_buildings, building_count, d_planes, grid, d_visible_indices, d_visible_count);
}

// wrapper for road frustum culling - simplified culling for linear features  
extern "C" void cuda_perform_road_culling_wrapper(gpu_road_data* d_roads, int road_count, float4* d_planes, int* d_visible_indices, int* d_visible_count) {
    // configure kernel launch parameters for road processing
    int block_size = BLOCK_SIZE; // optimal block size for gpu occupancy
    int grid_size = (road_count + block_size - 1) / block_size; // ceiling division for full coverage
    // initialize visible count to zero for atomic accumulation
    CUDA_CHECK(cudaMemset(d_visible_count, 0, sizeof(int)));
    // launch kernel with calculated parameters
    cuda_frustum_cull_roads<<<grid_size, block_size>>>(d_roads, d_planes, d_visible_indices, d_visible_count);
    // ensure kernel execution completed without errors
    CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());
}

// wrapper for spatial grid creation - organizes buildings for efficient culling
extern "C" void cuda_create_spatial_grid_wrapper(gpu_building_data* d_buildings, int building_count, cuda_grid* grid, float min_x, float min_z, float max_x, float max_z, float3* d_vertices) {
    // call specialized building grid creation function
    cuda_create_spatial_grid_for_buildings(d_buildings, building_count, grid, min_x, min_z, max_x, max_z, d_vertices);
}