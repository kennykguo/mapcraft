#ifndef _3D_RENDERING_CUH_
#define _3D_RENDERING_CUH_

#include <cuda_runtime.h>
#include <vector_types.h>
#include "cuda_globals.cuh"

// Forward declare OSMID if needed
struct OSMID;

// Helper function to compare OSMID values - can be executed both on HOST - cpu and DEVICE - gpu
__host__ __device__ inline uint64_t get_osmid_value(const OSMID& id);
__host__ __device__ inline bool osmid_less(const OSMID& a, const OSMID& b);
__host__ __device__ inline bool osmid_greater(const OSMID& a, const OSMID& b);

// Create and initialize a spatial grid for buildings
extern "C" void cuda_create_spatial_grid_for_buildings(
    gpu_building_data* d_buildings,
    int building_count,
    cuda_grid* grid,
    float min_x, float min_z, float max_x, float max_z,
    float3* d_vertices
);

// Perform frustum culling for buildings using spatial grid
extern "C" void cuda_perform_frustum_culling(
    gpu_building_data* d_buildings,
    int building_count,
    float4* d_planes,
    const cuda_grid& grid,
    int* d_visible_indices,
    int* d_visible_count
);

// Perform simple frustum culling for roads (no spatial grid)
extern "C" void cuda_perform_road_culling(
    gpu_road_data* d_roads,
    int road_count,
    float4* d_planes,
    int* d_visible_indices,
    int* d_visible_count
);

// Natural feature culling
extern "C" void cuda_perform_natural_feature_culling(
    gpu_natural_feature_data* d_natural_feature_data,
    int natural_feature_count,
    float4* d_frustum_planes,
    int* d_visible_indices,
    int* d_visible_count
);

// Check if a building is visible in the frustum
__device__ bool is_building_visible(
    const gpu_building_data& building,
    const float4* planes
);

// Check if a road is visible in the frustum
__device__ bool is_road_visible(
    const gpu_road_data& road,
    const float4* planes
);

// Compute the signed distance from a point to a plane
__device__ float signed_distance(
    const float3& point,
    const float4& plane
);

// Check if a sphere is visible in the frustum
__device__ bool is_sphere_visible(
    const float3& center,
    float radius,
    const float4* planes
);

#endif