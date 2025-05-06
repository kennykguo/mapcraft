#ifndef CUDA_GLOBALS_CUH_
#define CUDA_GLOBALS_CUH_
#include <cuda_runtime.h>
#include <vector_types.h>

// Maximum number of vertices per building to avoid memory issues
#define MAX_VERTICES 100

// Default cell size for the spatial grid (in meters)
#define GRID_CELL_SIZE 50.0f

// GPU representation of building data
struct gpu_building_data {
    int vertex_count; // Number of vertices
    int vertex_offset; // Offset into global vertex array
    float height; // Building height
    float3 centroid; // Center point
    float bounding_radius; // Radius of bounding sphere
    bool has_roof; // Whether the building has a distinct roof
    int roof_type; // 0=flat, 1=gabled, 2=hipped, etc.
    float3 color; // Pre-calculated color for consistent rendering
};

// GPU representation of road data
struct gpu_road_data {
    int vertex_count; // Number of vertices
    int vertex_offset; // Offset into global vertex array
    float width; // Width of the road
    float elevation; // Height above ground
    float3 centroid; // Center point
    float bounding_radius; // Radius of bounding sphere
    int road_type; // Type of road (0=highway, 1=primary, etc.)
};

// GPU structure for natural features
struct gpu_natural_feature_data {
    int vertex_count; // Number of vertices
    int vertex_offset; // Offset into the vertices array
    float elevation; // Height above/below ground
    float3 centroid; // Center point of the feature
    float bounding_radius; // Radius of bounding sphere for culling
    int feature_type; // Type of feature (0=lake, 1=river, 2=beach, etc.)
    float3 color; // Color for rendering
};

// Structure for a 2D spatial grid
struct cuda_grid {
    float2 origin; // Grid origin (min x, min z)
    float cell_size; // Cell size in meters
    int2 dimensions; // Number of cells in x and z directions
    int2* cell_ranges; // Start and end indices in the building_indices array for each cell
    int* building_indices; // Indices of buildings, sorted by cell
};

#endif // CUDA_GLOBALS_CUH_