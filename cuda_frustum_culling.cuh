#ifndef _CUDA_FRUSTUM_CULLING_CUH_
#define _CUDA_FRUSTUM_CULLING_CUH_

#include <cuda_runtime.h>
#include <vector_types.h>
#include "cuda_globals.cuh"
#include "cuda_spatial_grid.cuh"

// perform frustum culling on the GPU using spatial grid
extern "C" void cuda_perform_frustum_culling(
    gpu_building_data* d_buildings,
    int building_count,
    float4* d_planes,
    const cuda_grid& grid,
    int* d_visible_indices,
    int* d_visible_count
);

// perform simple frustum culling for roads (no spatial grid)
extern "C" void cuda_perform_road_culling(
    gpu_road_data* d_roads,
    int road_count,
    float4* d_planes,
    int* d_visible_indices,
    int* d_visible_count
);

// create frustum planes from view and projection matrices
extern "C" void cuda_create_frustum_planes(
    const float* view_matrix,
    const float* proj_matrix,
    float4* d_planes
);

// cuda kernels declarations
__global__ void calculate_frustum_planes_kernel(
    const float* view_matrix,
    const float* proj_matrix,
    float4* planes
);
__global__ void cuda_frustum_cull_buildings(
    const gpu_building_data* buildings,
    const float4* planes,
    const cuda_grid grid,
    int* visible_indices,
    int* visible_count
);
__global__ void cuda_frustum_cull_roads(
    const gpu_road_data* roads,
    const float4* planes,
    int* visible_indices,
    int* visible_count
);

// utility functions
__device__ float dot(const float3& a, const float3& b);
__device__ float signed_distance(const float3& point, const float4& plane);
__device__ bool is_sphere_visible(const float3& center, float radius, const float4* planes);
__device__ bool is_point_visible(const float3& point, const float4* planes);
__device__ void normalize_plane(float4& plane);

#endif
