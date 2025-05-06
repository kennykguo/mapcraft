#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include "cuda_globals.cuh"
#include "cuda_frustum_culling.cuh"
#include "cuda_spatial_grid.cuh"

// error checking macro - ensures immediate detection of cuda failures
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("cuda error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); } }

// compute dot product of two 3d vectors - fundamental linear algebra operation
__device__ float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// compute signed distance from a point to a plane - determines which side of plane
__device__ float signed_distance(const float3& point, const float4& plane) {
    // plane equation: ax + by + cz + d = 0
    return dot(make_float3(plane.x, plane.y, plane.z), point) + plane.w;
}

// check if a sphere is visible in the frustum - core culling test
__device__ bool is_sphere_visible(const float3& center, float radius, const float4* planes) {
    // test against all 6 frustum planes (left, right, top, bottom, near, far)
    for (int i = 0; i < 6; i++) {
        float distance = signed_distance(center, planes[i]);
        // if sphere center is farther than radius outside plane, it's invisible
        if (distance < -radius) {return false;}
    }
    return true; // sphere intersects or is inside all planes
}

// check if a point is visible in the frustum - simpler test for point geometry
__device__ bool is_point_visible(const float3& point, const float4* planes) {
    for (int i = 0; i < 6; i++) {
        // point must be on positive side of all planes to be visible
        if (signed_distance(point, planes[i]) < 0) {return false;}
    }
    return true; // point is inside all planes
}

// normalize a plane equation - converts to standard form for consistent testing
__device__ void normalize_plane(float4& plane) {
    // calculate magnitude of normal vector
    float magnitude = sqrtf(plane.x * plane.x + plane.y * plane.y + plane.z * plane.z);
    // divide all components by magnitude to get unit vector
    plane.x /= magnitude;plane.y /= magnitude;plane.z /= magnitude;plane.w /= magnitude;
}

// frustum culling kernel for buildings using spatial grid - leverages spatial acceleration
__global__ void cuda_frustum_cull_buildings(const gpu_building_data* buildings, const float4* planes, const cuda_grid grid, int* visible_indices, int* visible_count) {
    // load frustum planes into shared memory for fast access by all threads
    extern __shared__ float4 shared_planes[]; // dynamically sized shared memory
    if (threadIdx.x < 6) {shared_planes[threadIdx.x] = planes[threadIdx.x];}
    __syncthreads(); // ensure all threads see loaded planes
    
    // get cell index this thread is responsible for
    int cell_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell_idx >= grid.dimensions.x * grid.dimensions.y) return;
    
    // get range of buildings in this spatial cell - O(1) lookup
    int2 cell_range = grid.cell_ranges[cell_idx];
    
    // process each building in the cell - parallel spatial partitioning
    for (int i = cell_range.x; i < cell_range.y; i++) {
        int building_idx = grid.building_indices[i];
        const gpu_building_data& building = buildings[building_idx];
        
        // check if building is visible using sphere bound test
        if (is_sphere_visible(building.centroid, building.bounding_radius, shared_planes)) {
            // atomically add to visible list - thread-safe result collection
            int pos = atomicAdd(visible_count, 1);
            visible_indices[pos] = building_idx;}
    }
}

// simple frustum culling kernel for roads - no spatial acceleration needed
__global__ void cuda_frustum_cull_roads(const gpu_road_data* roads, const float4* planes, int* visible_indices, int* visible_count) {
    int road_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (road_idx >= blockDim.x * gridDim.x) return;
    
    // check if road is visible using sphere bound test
    if (is_sphere_visible(roads[road_idx].centroid, roads[road_idx].bounding_radius, planes)) {
        int pos = atomicAdd(visible_count, 1);
        visible_indices[pos] = road_idx;}
}

// kernel to calculate frustum planes from view/projection matrices - view frustum extraction 
__global__ void calculate_frustum_planes_kernel(const float* view_matrix, const float* proj_matrix, float4* planes) {
    // only need single thread for matrix calculation
    if (threadIdx.x > 0 || blockIdx.x > 0) return;
    
    // compute view-projection matrix - combines camera transform with perspective
    float vp[16];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 4; k++) {
                sum += proj_matrix[i * 4 + k] * view_matrix[k * 4 + j];}
            vp[i * 4 + j] = sum;}
    }
    
    // extract frustum planes from view-projection matrix using standard equations
    // left plane: normalize(vp[3] + vp[0]) 
    planes[0].x = vp[3] + vp[0];planes[0].y = vp[7] + vp[4];planes[0].z = vp[11] + vp[8];planes[0].w = vp[15] + vp[12];
    normalize_plane(planes[0]);
    
    // right plane: normalize(vp[3] - vp[0])
    planes[1].x = vp[3] - vp[0];planes[1].y = vp[7] - vp[4];planes[1].z = vp[11] - vp[8];planes[1].w = vp[15] - vp[12];
    normalize_plane(planes[1]);
    
    // bottom plane: normalize(vp[3] + vp[1])
    planes[2].x = vp[3] + vp[1];planes[2].y = vp[7] + vp[5];planes[2].z = vp[11] + vp[9];planes[2].w = vp[15] + vp[13];
    normalize_plane(planes[2]);
    
    // top plane: normalize(vp[3] - vp[1])
    planes[3].x = vp[3] - vp[1];planes[3].y = vp[7] - vp[5];planes[3].z = vp[11] - vp[9];planes[3].w = vp[15] - vp[13];
    normalize_plane(planes[3]);
    
    // near plane: normalize(vp[3] + vp[2])
    planes[4].x = vp[3] + vp[2];planes[4].y = vp[7] + vp[6];planes[4].z = vp[11] + vp[10];planes[4].w = vp[15] + vp[14];
    normalize_plane(planes[4]);
    
    // far plane: normalize(vp[3] - vp[2])
    planes[5].x = vp[3] - vp[2];planes[5].y = vp[7] - vp[6];planes[5].z = vp[11] - vp[10];planes[5].w = vp[15] - vp[14];
    normalize_plane(planes[5]);
}

// host function to extract frustum planes from matrices
extern "C" void cuda_create_frustum_planes(const float* view_matrix, const float* proj_matrix, float4* d_planes) {
    // single thread launch for matrix calculation
    calculate_frustum_planes_kernel<<<1, 1>>>(view_matrix, proj_matrix, d_planes);
    CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());
}

// host function to perform frustum culling with spatial acceleration
extern "C" void cuda_perform_frustum_culling(gpu_building_data* d_buildings, int building_count, float4* d_planes, const cuda_grid& grid, int* d_visible_indices, int* d_visible_count) {
    // initialize visible count to zero
    CUDA_CHECK(cudaMemset(d_visible_count, 0, sizeof(int)));
    
    // calculate kernel launch parameters for spatial cells
    int num_cells = grid.dimensions.x * grid.dimensions.y;
    int block_size = 256;int grid_size = (num_cells + block_size - 1) / block_size;
    
    // allocate shared memory for frustum planes - 6 planes
    size_t shared_mem_size = 6 * sizeof(float4);
    
    // launch kernel with dynamic shared memory
    cuda_frustum_cull_buildings<<<grid_size, block_size, shared_mem_size>>>(d_buildings, d_planes, grid, d_visible_indices, d_visible_count);
    CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());
}

// host function to perform simple frustum culling for roads
extern "C" void cuda_perform_road_culling(gpu_road_data* d_roads, int road_count, float4* d_planes, int* d_visible_indices, int* d_visible_count) {
    CUDA_CHECK(cudaMemset(d_visible_count, 0, sizeof(int)));
    
    // calculate kernel launch parameters for roads
    int block_size = 256;int grid_size = (road_count + block_size - 1) / block_size;
    
    // launch simple culling kernel without spatial acceleration
    cuda_frustum_cull_roads<<<grid_size, block_size>>>(d_roads, d_planes, d_visible_indices, d_visible_count);
    CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());
}

// kernel for natural feature culling - handles environmental elements
__global__ void natural_feature_culling_kernel(gpu_natural_feature_data* natural_feature_data, int natural_feature_count, float4* frustum_planes, int* visible_indices, int* visible_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= natural_feature_count) return;
    
    const gpu_natural_feature_data& feature = natural_feature_data[idx];
    
    // perform sphere-frustum intersection test
    bool is_visible = true;
    for (int i = 0; i < 6; i++) {
        const float4& plane = frustum_planes[i];
        // calculate signed distance from feature center to plane
        float dist = plane.x * feature.centroid.x + plane.y * feature.centroid.y + plane.z * feature.centroid.z + plane.w;
        
        // if entire sphere is on negative side of plane, it's invisible  
        if (dist < -feature.bounding_radius) {is_visible = false;break;}
    }
    
    // atomically add visible features to result list
    if (is_visible) {
        int idx_in_visible = atomicAdd(visible_count, 1);
        visible_indices[idx_in_visible] = idx;}
}

// host function to cull natural features
extern "C" void cuda_perform_natural_feature_culling(gpu_natural_feature_data* d_natural_feature_data, int natural_feature_count, float4* d_frustum_planes, int* d_visible_indices, int* d_visible_count) {
    if (natural_feature_count <= 0) return;
    
    // calculate kernel launch parameters
    int block_size = 256;
    int grid_size = (natural_feature_count + block_size - 1) / block_size;
    
    // launch kernel for natural feature culling
    natural_feature_culling_kernel<<<grid_size, block_size>>>(d_natural_feature_data, natural_feature_count, d_frustum_planes, d_visible_indices, d_visible_count);
    
    // check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {fprintf(stderr, "cuda error in natural feature culling kernel: %s\n", cudaGetErrorString(err));}
}