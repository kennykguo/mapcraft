#include <3d_rendering.h>

// uploads building data to gpu if buildings exist
// allocates memory for visible building indices and count
// uploads road segments to gpu if roads exist
// allocates memory for visible road indices and count
// uploads natural features to gpu if natural features exist
// allocates memory for visible natural feature indices and count
// initializes spatial grid structure for buildings
// uses error checking for cuda memory operations
void Renderer3D::init_cuda_resources() {
    std::cout << "initializing cuda resources..." << std::endl;
    // check if we have data to process
    if (buildings.empty() && road_segments.empty() && natural_features.empty()) {
        std::cout << "No data to process, skipping CUDA initialization" << std::endl;
        return;
    }
    
    // upload buildings to gpu
    if (!buildings.empty()) {
        upload_buildings_to_gpu();
        // allocate memory for visible building indices and count
        if (!d_visible_building_indices) {
            cuda_check(cudaMalloc(&d_visible_building_indices, buildings.size() * sizeof(int)), "Allocate visible building indices");
        }
        if (!d_visible_building_count) {
            cuda_check(cudaMalloc(&d_visible_building_count, sizeof(int)), "Allocate visible building count");
        }
    }
    
    // upload roads to gpu
    if (!road_segments.empty()) {
        upload_roads_to_gpu();
        // allocate memory for visible road indices and count
        if (!d_visible_road_indices) {
            cuda_check(cudaMalloc(&d_visible_road_indices, road_segments.size() * sizeof(int)), "Allocate visible road indices");
        }
        if (!d_visible_road_count) {
            cuda_check(cudaMalloc(&d_visible_road_count, sizeof(int)), "Allocate visible road count");
        }
    }
    
    // upload natural features to gpu
    if (!natural_features.empty()) {
        upload_natural_features_to_gpu();
        // memory for natural features is allocated in upload_natural_features_to_gpu
        if (!d_visible_natural_feature_indices) {
            cuda_check(cudaMalloc(&d_visible_natural_feature_indices, natural_features.size() * sizeof(int)), "Allocate visible natural feature indices");
        }
        if (!d_visible_natural_feature_count) {
            cuda_check(cudaMalloc(&d_visible_natural_feature_count, sizeof(int)), "Allocate visible natural feature count");
        }
    }
    
    // initialize spatial grid for buildings
    init_spatial_grid();
    std::cout << "CUDA resources initialized" << std::endl;
}

// HELPER FUNCTIONS FOR init_cuda_resources
// upload building data to gpu
void Renderer3D::upload_buildings_to_gpu() {
    // validate cuda function before attempting
    std::cout << "uploading buildings to GPU..." << std::endl;
    cuda_check(cudaGetLastError(), "Before building upload");
    
    // allocate gpu memory for building data
    gpu_building_data* h_building_data = new gpu_building_data[buildings.size()];
    
    // prepare building data for upload
    for (size_t i = 0; i < buildings.size(); i++) {
        h_building_data[i].vertex_count = buildings[i].vertex_count;
        h_building_data[i].height = buildings[i].height;
        h_building_data[i].centroid = make_float3(
            buildings[i].centroid.x,
            buildings[i].centroid.y,
            buildings[i].centroid.z
        );
        h_building_data[i].bounding_radius = buildings[i].bounding_radius;
        h_building_data[i].has_roof = buildings[i].has_roof;
        h_building_data[i].roof_type = buildings[i].roof_type;
        h_building_data[i].vertex_offset = 0;
    }
    size_t total_vertices = 0;
    for (size_t i = 0; i < buildings.size(); i++) {
        h_building_data[i].vertex_offset = total_vertices;
        total_vertices += buildings[i].vertex_count;
    }

    cuda_check(cudaMalloc(&d_building_data, buildings.size() * sizeof(gpu_building_data)),  "Allocate building data"); // allocate gpu memory for building data
    cuda_check(cudaMalloc(&d_building_vertices, total_vertices * sizeof(float3)),  "Allocate building vertices"); // allocate gpu memory for vertices
    cuda_check(cudaMemcpy(d_building_data, h_building_data, buildings.size() * sizeof(gpu_building_data), cudaMemcpyHostToDevice), "Copy building data to GPU"); // upload building data
    
    // prepare and upload building vertices
    float3* h_vertices = new float3[total_vertices];
    size_t vertex_idx = 0;
    for (const auto& building : buildings) {
        for (int i = 0; i < building.vertex_count; i++) {
            h_vertices[vertex_idx].x = building.vertices[i].x;
            h_vertices[vertex_idx].y = building.vertices[i].y;
            h_vertices[vertex_idx].z = building.vertices[i].z;
            vertex_idx++;
        }
    }
    
    cuda_check(cudaMemcpy(d_building_vertices, h_vertices, total_vertices * sizeof(float3),  cudaMemcpyHostToDevice), "Copy building vertices to GPU");
    
    // cleanup temporary host arrays
    delete[] h_building_data;
    delete[] h_vertices;
    
    // set building count for use in kernels
    building_count = buildings.size();

    std::cout << "Buildings uploaded to GPU: " << building_count << " buildings, " 
              << total_vertices << " vertices" << std::endl;
}

// upload road data to gpu
void Renderer3D::upload_roads_to_gpu() {
    std::cout << "Uploading roads to GPU..." << std::endl;
    
    // validate cuda function before attempting
    cuda_check(cudaGetLastError(), "Before road upload");
    
    // allocate gpu memory for road data
    gpu_road_data* h_road_data = new gpu_road_data[road_segments.size()];
    
    // prepare road data for upload
    for (size_t i = 0; i < road_segments.size(); i++) {
        h_road_data[i].vertex_count = road_segments[i].vertex_count;
        h_road_data[i].width = road_segments[i].width;
        h_road_data[i].elevation = road_segments[i].elevation;
        h_road_data[i].centroid = make_float3(
            road_segments[i].centroid.x,
            road_segments[i].centroid.y,
            road_segments[i].centroid.z
        );
        h_road_data[i].bounding_radius = road_segments[i].bounding_radius;
        h_road_data[i].road_type = road_segments[i].road_type;
        h_road_data[i].vertex_offset = 0;
    }
    size_t total_vertices = 0;
    for (size_t i = 0; i < road_segments.size(); i++) {
        h_road_data[i].vertex_offset = total_vertices;
        total_vertices += road_segments[i].vertex_count;
    }
    
    cuda_check(cudaMalloc(&d_road_data, road_segments.size() * sizeof(gpu_road_data)),  "Allocate road data"); // allocate gpu memory for road data
    cuda_check(cudaMalloc(&d_road_vertices, total_vertices * sizeof(float3)), "Allocate road vertices");   // allocate gpu memory for vertices
    cuda_check(cudaMemcpy(d_road_data, h_road_data,  road_segments.size() * sizeof(gpu_road_data),  cudaMemcpyHostToDevice),  "Copy road data to GPU");
    
    // prepare and upload vertices
    float3* h_vertices = new float3[total_vertices];
    size_t vertex_idx = 0;
    for (const auto& road : road_segments) {
        for (int i = 0; i < road.vertex_count; i++) {
            h_vertices[vertex_idx].x = road.vertices[i].x;
            h_vertices[vertex_idx].y = road.vertices[i].y;
            h_vertices[vertex_idx].z = road.vertices[i].z;
            vertex_idx++;
        }
    }
    
    cuda_check(cudaMemcpy(d_road_vertices, h_vertices, 
                         total_vertices * sizeof(float3), 
                         cudaMemcpyHostToDevice), 
              "Copy road vertices to GPU");
    
    // cleanup temporary host arrays
    delete[] h_road_data;
    delete[] h_vertices;
    
    // set road count for use in kernels
    road_count = road_segments.size();
    
    std::cout << "Roads uploaded to GPU: " << road_count << " roads, " 
              << total_vertices << " vertices" << std::endl;
}

// upload natural features to gpu
void Renderer3D::upload_natural_features_to_gpu() {
    std::cout << "Uploading natural features to GPU..." << std::endl;
    
    // skip if no natural features
    if (natural_features.empty()) {
        std::cout << "No natural features to upload" << std::endl;
        return;
    }
    // validate cuda function before attempting
    cuda_check(cudaGetLastError(), "Before natural feature upload");
    
    // allocate gpu memory for natural feature data
    gpu_natural_feature_data* h_natural_feature_data = new gpu_natural_feature_data[natural_features.size()];
    
    // map feature types to integer codes for gpu
    std::map<std::string, int> type_map = {
        {"lake", 0},
        {"river", 1},
        {"stream", 2},
        {"beach", 3},
        {"greenspace", 4},
        {"park", 5}
    };
    
    // prepare natural feature data for upload
    for (size_t i = 0; i < natural_features.size(); i++) {
        h_natural_feature_data[i].vertex_count = natural_features[i].vertex_count;
        h_natural_feature_data[i].elevation = natural_features[i].elevation;
        h_natural_feature_data[i].centroid = make_float3(
            natural_features[i].centroid.x,
            natural_features[i].centroid.y,
            natural_features[i].centroid.z
        );
        h_natural_feature_data[i].bounding_radius = natural_features[i].bounding_radius;
        // map feature type string to integer code
        h_natural_feature_data[i].feature_type = type_map.count(natural_features[i].type) ?
            type_map[natural_features[i].type] : 99; // use 99 for unknown types
        // set color
        h_natural_feature_data[i].color = make_float3(
            natural_features[i].color.x,
            natural_features[i].color.y,
            natural_features[i].color.z
        );
        // count total vertices needed
        h_natural_feature_data[i].vertex_offset = 0; // will be set in a second pass
    }
    
    // calculate vertex offsets
    size_t total_vertices = 0;
    for (size_t i = 0; i < natural_features.size(); i++) {
        h_natural_feature_data[i].vertex_offset = total_vertices;
        total_vertices += natural_features[i].vertex_count;
    }
    
    // allocate gpu memory for natural feature data
    cuda_check(cudaMalloc(&d_natural_feature_data, natural_features.size() * sizeof(gpu_natural_feature_data)), "Allocate natural feature data");
    
    // allocate gpu memory for vertices
    cuda_check(cudaMalloc(&d_natural_feature_vertices, total_vertices * sizeof(float3)), "Allocate natural feature vertices");
    
    // upload natural feature data
    cuda_check(cudaMemcpy(d_natural_feature_data, h_natural_feature_data, natural_features.size() * sizeof(gpu_natural_feature_data),  cudaMemcpyHostToDevice), "Copy natural feature data to GPU");
    
    // prepare and upload vertices
    float3* h_vertices = new float3[total_vertices];
    size_t vertex_idx = 0;
    for (const auto& feature : natural_features) {
        for (int i = 0; i < feature.vertex_count; i++) {
            h_vertices[vertex_idx].x = feature.vertices[i].x;
            h_vertices[vertex_idx].y = feature.vertices[i].y;
            h_vertices[vertex_idx].z = feature.vertices[i].z;
            vertex_idx++;
        }
    }
    
    cuda_check(cudaMemcpy(d_natural_feature_vertices, h_vertices,  total_vertices * sizeof(float3), cudaMemcpyHostToDevice), "Copy natural feature vertices to GPU");
    
    // cleanup temporary host arrays
    delete[] h_natural_feature_data;
    delete[] h_vertices;
    
    // set natural feature count
    natural_feature_count = natural_features.size();
    
    std::cout << "Natural features uploaded to GPU: " << natural_feature_count << " features, " 
              << total_vertices << " vertices" << std::endl;
}

// initialize spatial grid for buildings
void Renderer3D::init_spatial_grid() {
    if (building_count == 0) {
        std::cout << "no buildings to create spatial grid for" << std::endl;
        return;
    }
    
    std::cout << "Initializing spatial grid..." << std::endl;
    
    // calculate grid bounds based on building data - start with max as min #, and min as max #
    float min_x = std::numeric_limits<float>::max();
    float min_z = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::min();
    float max_z = std::numeric_limits<float>::min();
    
    for (const auto& building : buildings) {
        min_x = std::min(min_x, building.centroid.x - building.bounding_radius);
        min_z = std::min(min_z, building.centroid.z - building.bounding_radius);
        max_x = std::max(max_x, building.centroid.x + building.bounding_radius);
        max_z = std::max(max_z, building.centroid.z + building.bounding_radius);
    }
    
    // add some padding
    min_x -= 10.0f;
    min_z -= 10.0f;
    max_x += 10.0f;
    max_z += 10.0f;
    
    // create spatial grid
    cuda_create_spatial_grid_for_buildings(
        d_building_data,
        building_count,
        &d_grid,
        min_x, min_z, max_x, max_z,
        d_building_vertices
    );
    std::cout << "Spatial grid initialized" << std::endl;
}

// cleanup cuda resources
void Renderer3D::cleanup_cuda_resources() {
    std::cout << "Cleaning up CUDA resources..." << std::endl;
    // free building data
    if (d_building_data) {
        cuda_check(cudaFree(d_building_data), "Free building data");
        d_building_data = nullptr;
    }
    // free building vertices
    if (d_building_vertices) {
        cuda_check(cudaFree(d_building_vertices), "Free building vertices");
        d_building_vertices = nullptr;
    }
    // free road data
    if (d_road_data) {
        cuda_check(cudaFree(d_road_data), "Free road data");
        d_road_data = nullptr;
    }
    // free road vertices
    if (d_road_vertices) {
        cuda_check(cudaFree(d_road_vertices), "Free road vertices");
        d_road_vertices = nullptr;
    }
    // free natural feature data
    if (d_natural_feature_data) {
        cuda_check(cudaFree(d_natural_feature_data), "Free natural feature data");
        d_natural_feature_data = nullptr;
    }
    // free natural feature vertices
    if (d_natural_feature_vertices) {
        cuda_check(cudaFree(d_natural_feature_vertices), "Free natural feature vertices");
        d_natural_feature_vertices = nullptr;
    }
    // free visible indices arrays
    if (d_visible_building_indices) {
        cuda_check(cudaFree(d_visible_building_indices), "Free visible building indices");
        d_visible_building_indices = nullptr;
    }
    if (d_visible_building_count) {
        cuda_check(cudaFree(d_visible_building_count), "Free visible building count");
        d_visible_building_count = nullptr;
    }
    if (d_visible_road_indices) {
        cuda_check(cudaFree(d_visible_road_indices), "Free visible road indices");
        d_visible_road_indices = nullptr;
    }
    if (d_visible_road_count) {
        cuda_check(cudaFree(d_visible_road_count), "Free visible road count");
        d_visible_road_count = nullptr;
    }
    if (d_visible_natural_feature_indices) {
        cuda_check(cudaFree(d_visible_natural_feature_indices), "Free visible natural feature indices");
        d_visible_natural_feature_indices = nullptr;
    }
    if (d_visible_natural_feature_count) {
        cuda_check(cudaFree(d_visible_natural_feature_count), "Free visible natural feature count");
        d_visible_natural_feature_count = nullptr;
    }
    // free frustum planes
    if (d_frustum_planes) {
        cuda_check(cudaFree(d_frustum_planes), "Free frustum planes");
        d_frustum_planes = nullptr;
    }
    // free grid resources
    if (d_grid.cell_ranges) {
        cuda_check(cudaFree(d_grid.cell_ranges), "Free grid cell ranges");
        d_grid.cell_ranges = nullptr;
    }
    if (d_grid.building_indices) {
        cuda_check(cudaFree(d_grid.building_indices), "Free grid building indices");
        d_grid.building_indices = nullptr;
    }
    std::cout << "CUDA resources cleaned up" << std::endl;
}

// check cuda errors and throw exception if any
void Renderer3D::cuda_check(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(err));
    } 
}