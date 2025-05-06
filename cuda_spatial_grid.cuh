#ifndef _CUDA_SPATIAL_GRID_CUH_
#define _CUDA_SPATIAL_GRID_CUH_

#include <cuda_runtime.h>
#include <vector_types.h>
#include "cuda_globals.cuh"

// Create and initialize a spatial grid for buildings
extern "C" void cuda_create_spatial_grid_for_buildings(
    gpu_building_data* d_buildings,
    int building_count,
    cuda_grid* grid,
    float min_x, float min_z, float max_x, float max_z,
    float3* d_vertices
);

// Generic function to create a spatial grid (not used in current implementation)
extern "C" void cuda_create_spatial_grid(
    const void* data,
    int entity_count,
    cuda_grid* grid,
    float min_x, float min_z,
    float max_x, float max_z,
    float cell_size
);

// Free memory allocated for a spatial grid
extern "C" void free_cuda_grid(cuda_grid* grid);

// Utility functions
__device__ __host__ int2 get_cell_index(const cuda_grid& grid, float x, float z);
__device__ __host__ int2 get_cell_index(const cuda_grid& grid, const float3& position);
__device__ __host__ int get_flattened_cell_index(const cuda_grid& grid, int cell_x, int cell_z);
__device__ __host__ int2 get_cell_range(const cuda_grid& grid, int cell_x, int cell_z);

// CUDA kernel declarations
__global__ void count_buildings_per_cell_kernel(
    const gpu_building_data* buildings,
    int building_count,
    const cuda_grid grid,
    int* cell_counts
);

__global__ void assign_buildings_to_cells_kernel(
    const gpu_building_data* buildings,
    int building_count,
    const cuda_grid grid,
    int* cell_offsets,
    int* cell_counts,
    int* building_indices
);

__global__ void setup_cell_ranges_kernel(
    const int* cell_offsets,
    const int* cell_counts,
    int2* cell_ranges,
    int num_cells
);

#endif