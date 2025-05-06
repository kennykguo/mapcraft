#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <cmath>
#include <stdio.h>
#include <float.h>
#include "cuda_globals.cuh"
#include "cuda_spatial_grid.cuh"

// error checking macro - immediately catches and reports cuda errors
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("cuda error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); } }

// get cell index for a position - converts world coordinates to grid cell indices
__device__ __host__ int2 get_cell_index(const cuda_grid& grid, float x, float z) {
    // divide by cell size to get cell coordinates in infinite space
    int cell_x = (int)floor((x - grid.origin.x) / grid.cell_size);
    int cell_z = (int)floor((z - grid.origin.y) / grid.cell_size);
    
    // clamp to grid boundaries preventing out-of-bounds array access
    cell_x = max(0, min(grid.dimensions.x - 1, cell_x));
    cell_z = max(0, min(grid.dimensions.y - 1, cell_z));
    
    return make_int2(cell_x, cell_z);
}

// overloaded version for float3 positions - convenience function for 3d vectors
__device__ __host__ int2 get_cell_index(const cuda_grid& grid, const float3& position) {
    // note: using x,z as the horizontal plane for architectural/geographic layout
    return get_cell_index(grid, position.x, position.z);
}

// get flattened index for 2d grid - converts 2d cell coords to 1d array index
__device__ __host__ int get_flattened_cell_index(const cuda_grid& grid, int cell_x, int cell_z) {
    // row-major order: cell_z rows of cell_x elements each
    return cell_z * grid.dimensions.x + cell_x;
}

// get range of entities in a cell - retrieves start/end indices for cell's data
__device__ __host__ int2 get_cell_range(const cuda_grid& grid, int cell_x, int cell_z) {
    int flat_idx = get_flattened_cell_index(grid, cell_x, cell_z);
    // returns int2 where x=start index, y=end index
    return grid.cell_ranges[flat_idx];
}

// kernel to count buildings per cell - first pass of grid construction algorithm
__global__ void count_buildings_per_cell_kernel(const gpu_building_data* buildings, int building_count, const cuda_grid grid, int* cell_counts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // calculate global thread index
    if (idx >= building_count) return; // guard against out-of-bounds access
    
    // find which cell this building belongs to based on its centroid position
    int2 cell = get_cell_index(grid, buildings[idx].centroid);
    int cell_idx = get_flattened_cell_index(grid, cell.x, cell.y);
    
    // atomically increment count for this cell - thread-safe concurrent counter
    atomicAdd(&cell_counts[cell_idx], 1);
}

// kernel to assign buildings to cells - second pass populates the spatial grid
__global__ void assign_buildings_to_cells_kernel(const gpu_building_data* buildings, int building_count, const cuda_grid grid, int* cell_offsets, int* cell_counts, int* building_indices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= building_count) return;
    
    // determine which cell this building belongs to
    int2 cell = get_cell_index(grid, buildings[idx].centroid);
    int cell_idx = get_flattened_cell_index(grid, cell.x, cell.y);
    
    // calculate where to store this building in the sorted array
    // offset gives starting position, atomic add returns next available slot
    int pos = cell_offsets[cell_idx] + atomicAdd(&cell_counts[cell_idx], 1);
    
    // store the building index at its sorted position
    building_indices[pos] = idx;
}

// kernel to set up cell ranges - final setup creates lookup structure
__global__ void setup_cell_ranges_kernel(const int* cell_offsets, const int* cell_counts, int2* cell_ranges, int num_cells) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cells) return;
    
    // create range [start, end) for each cell's buildings
    // allows O(1) lookup of all buildings in a cell
    cell_ranges[idx] = make_int2(cell_offsets[idx], cell_offsets[idx] + cell_counts[idx]);
}

// create spatial grid for buildings on gpu - main entry point for grid construction
extern "C" void cuda_create_spatial_grid_for_buildings(gpu_building_data* d_buildings, int building_count, cuda_grid* grid, float min_x, float min_z, float max_x, float max_z, float3* d_vertices) {
    if (building_count == 0) {printf("no buildings to create grid for\n"); return;}
    
    // initialize grid properties - defines the spatial partitioning space
    grid->origin = make_float2(min_x, min_z);
    grid->cell_size = GRID_CELL_SIZE;
    
    // calculate grid dimensions needed to cover all data
    int dim_x = (int)ceil((max_x - min_x) / GRID_CELL_SIZE);
    int dim_z = (int)ceil((max_z - min_z) / GRID_CELL_SIZE);
    dim_x = max(1, dim_x); dim_z = max(1, dim_z); // ensure minimum 1x1 grid
    grid->dimensions = make_int2(dim_x, dim_z);
    
    int num_cells = dim_x * dim_z;
    printf("creating spatial grid with %d x %d = %d cells\n", dim_x, dim_z, num_cells);
    
    // allocate device memory for grid data structures
    CUDA_CHECK(cudaMalloc(&grid->cell_ranges, num_cells * sizeof(int2)));
    int* d_cell_counts;CUDA_CHECK(cudaMalloc(&d_cell_counts, num_cells * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_cell_counts, 0, num_cells * sizeof(int)));
    
    // calculate grid size for kernel launches - ensures all buildings are processed
    int block_size = 256; // standard block size for good gpu occupancy
    int grid_size = (building_count + block_size - 1) / block_size;
    
    // count buildings per cell - parallel histogram construction
    count_buildings_per_cell_kernel<<<grid_size, block_size>>>(d_buildings, building_count, *grid, d_cell_counts);
    CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());
    
    // calculate prefix sum for cell offsets - determines where each cell's data starts
    int* d_cell_offsets;CUDA_CHECK(cudaMalloc(&d_cell_offsets, num_cells * sizeof(int)));
    thrust::device_ptr<int> dev_counts(d_cell_counts);
    thrust::device_ptr<int> dev_offsets(d_cell_offsets);
    thrust::exclusive_scan(dev_counts, dev_counts + num_cells, dev_offsets);
    
    // verify total count - ensures no buildings were lost in processing
    int total_buildings_in_grid;int last_count;
    CUDA_CHECK(cudaMemcpy(&last_count, &d_cell_counts[num_cells - 1], sizeof(int), cudaMemcpyDeviceToHost));
    int last_offset;
    CUDA_CHECK(cudaMemcpy(&last_offset, &d_cell_offsets[num_cells - 1], sizeof(int), cudaMemcpyDeviceToHost));
    total_buildings_in_grid = last_offset + last_count;
    
    if (total_buildings_in_grid != building_count) {
        printf("warning: building count mismatch. expected %d, got %d\n", building_count, total_buildings_in_grid);}
    
    // allocate and populate building indices array - sorted by cell
    CUDA_CHECK(cudaMalloc(&grid->building_indices, building_count * sizeof(int)));
    int* d_temp_counts;CUDA_CHECK(cudaMalloc(&d_temp_counts, num_cells * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_temp_counts, 0, num_cells * sizeof(int)));
    
    // assign buildings to cells - creates final sorted structure
    assign_buildings_to_cells_kernel<<<grid_size, block_size>>>(d_buildings, building_count, *grid, d_cell_offsets, d_temp_counts, grid->building_indices);
    CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());
    
    // set up cell ranges - creates fast lookup table
    int cells_grid_size = (num_cells + block_size - 1) / block_size;
    setup_cell_ranges_kernel<<<cells_grid_size, block_size>>>(d_cell_offsets, d_cell_counts, grid->cell_ranges, num_cells);
    CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());
    
    // cleanup temporary memory to prevent leaks
    CUDA_CHECK(cudaFree(d_cell_counts));CUDA_CHECK(cudaFree(d_cell_offsets));CUDA_CHECK(cudaFree(d_temp_counts));
    printf("spatial grid created successfully\n");
}

// free memory allocated for spatial grid - essential cleanup function
extern "C" void free_cuda_grid(cuda_grid* grid) {
    if (grid->cell_ranges) {CUDA_CHECK(cudaFree(grid->cell_ranges)); grid->cell_ranges = nullptr;}
    if (grid->building_indices) {CUDA_CHECK(cudaFree(grid->building_indices)); grid->building_indices = nullptr;}
}

// generic function to create a spatial grid on gpu - template for other entity types
extern "C" void cuda_create_spatial_grid(const void* data, int entity_count, cuda_grid* grid, float min_x, float min_z, float max_x, float max_z, float cell_size) {
    // this is a placeholder showing the interface pattern for different entity types
    printf("warning: generic cuda_create_spatial_grid called. this is a placeholder function.\n");
    printf("to create a grid for buildings, use cuda_create_spatial_grid_for_buildings instead.\n");
    
    // initialize grid properties for future implementation
    grid->origin = make_float2(min_x, min_z);
    grid->cell_size = cell_size;
    
    // calculate dimensions for any entity type
    int dim_x = (int)ceil((max_x - min_x) / cell_size);
    int dim_z = (int)ceil((max_z - min_z) / cell_size);
    dim_x = max(1, dim_x); dim_z = max(1, dim_z); // ensure minimum 1x1 grid
    grid->dimensions = make_int2(dim_x, dim_z);
    
    // initialize to null as actual grid not created
    grid->cell_ranges = nullptr;
    grid->building_indices = nullptr;
}