#include <3d_rendering.h>

// constant for conversion and distance calculations
const float WATER_ELEVATION = -0.5f;
const float DEFAULT_ELEVATION = 0.1f;
// define which feature types count as natural for landscape rendering
const std::vector<std::string> natural_types = {"park", "water", "river", "lake", "wood", "grass", "beach"};

// load building data from host to device
void Renderer3D::load_features() {
  std::lock_guard<std::mutex> lock(map_data_mutex);  // ensure thread-safe access to shared data
  std::cout << "Loading feature data to GPU..." << std::endl;
  glm::vec3 reference_3d = latlon_to_3d(reference_point, reference_point); // convert reference point to 3d coordinates
  buildings.clear(); // remove any existing building data
  buildings.reserve(graphics_feature_data.size()); // allocate memory (more than enough)

  int building_count = 0, rejected = 0, non_building = 0;

  for (const auto& feature : graphics_feature_data) {
      if (feature.type == "building") {  // filter for building features only
          // calculate straight-line distance from reference point in the xz plane (ignoring height)
          float distance = glm::distance(glm::vec2(feature.centroid3d.x, feature.centroid3d.z), glm::vec2(reference_3d.x, reference_3d.z));
          if (distance <= render_radius && feature.points3d.size() > 0) {  // check if building is within render distance and has valid # of points
                building building; // create new building structure
                building.vertex_count = feature.points3d.size(); // set number of vertices
                building.vertices = new glm::vec3[building.vertex_count]; // allocate array for vertex positions
                for (int i = 0; i < building.vertex_count; i++){
                building.vertices[i] = feature.points3d[i];
                } 
                building.height = feature.height;  // building height in meters
                building.centroid = feature.centroid3d;  // center point for culling and positioning
                building.bounding_radius = feature.bounding_radius;  // radius for collision detection
                building.has_roof = feature.has_roof;  // whether to render a roof
                building.roof_type = feature.roof_type;  // roof geometry type
                building.color = color_to_vec3(feature.rgba);  // convert rgba to rgb
                buildings.push_back(building);  // add to building list
                building_count++;
          } else {
              rejected++;
          }
      } else {
          non_building++;
      }
  }

  std::cout << "Loaded " << building_count << " buildings (" << rejected << " rejected, " << non_building << " non-buildings)\n";
}

// load road segment data for rendering streets and paths
void Renderer3D::load_roads() {
  std::lock_guard<std::mutex> lock(map_data_mutex);  // thread-safe data access
  std::cout << "Loading road data to GPU..." << std::endl;
  
  road_segments.clear();  // clear existing road data
  road_segments.reserve(graphics_street_segment_data.size());  // optimize memory allocation
  glm::vec3 reference_3d = latlon_to_3d(reference_point, reference_point);  // get 3d reference point

  for (const auto& segment : graphics_street_segment_data) {
        // check distance from reference point using 2d projection
        float distance = glm::distance(glm::vec2(segment.centroid3d.x, segment.centroid3d.z), glm::vec2(reference_3d.x, reference_3d.z));
        if (distance <= render_radius) {  // only load roads within render distance
            road_segment road;  // create road structure
            road.vertex_count = segment.points3d.size();  // number of points defining the road
            road.vertices = new glm::vec3[road.vertex_count];  // allocate vertex array
            // copy vertex positions
            for (int i = 0; i < road.vertex_count; i++){
            road.vertices[i] = segment.points3d[i];
            }
            road.width = segment.width;  // road width in meters for rendering
            road.elevation = segment.height;  // elevation offset from terrain
            road.centroid = segment.centroid3d;  // center point for culling
            road.bounding_radius = segment.bounding_radius;  // spatial bounds
            // classify road by type for different rendering styles
            road.road_type = segment.highway_type == "motorway" ? 0 :  // highways
                            segment.highway_type == "primary" ? 1 :     // main roads
                            segment.highway_type == "secondary" ? 2 :   // side streets
                            segment.highway_type == "residential" ? 3 : // neighborhood streets
                            4;  // footpaths and others
            road_segments.push_back(road);  // add to road vector
        }
  }
  std::cout << "Loaded " << road_segments.size() << " road segments\n";
}

// load parks, water bodies, and other natural landscape features
void Renderer3D::load_natural_features() {
  std::lock_guard<std::mutex> lock(map_data_mutex);  // thread synchronization
  std::cout << "Loading natural features..." << std::endl;
  
  natural_features.clear();  // clear existing natural features
  glm::vec3 reference_3d = latlon_to_3d(reference_point, reference_point);  // 3d reference for distance

  for (const auto& feature : graphics_feature_data) {
        std::string type = feature.type;  // get feature type
        std::transform(type.begin(), type.end(), type.begin(), ::tolower);  // convert to lowercase for comparison
        
        // check if feature type matches any natural category
        bool is_natural = std::any_of(natural_types.begin(), natural_types.end(), 
                                    [&type](const std::string& natural_type) { 
                                        return type.find(natural_type) != std::string::npos; 
                                    });

        if (is_natural) {  // process natural features
            // calculate distance from reference point
            float distance = glm::distance(glm::vec2(feature.centroid3d.x, feature.centroid3d.z), glm::vec2(reference_3d.x, reference_3d.z));

            if (distance <= render_radius && feature.points3d.size() >= 3) {  // need at least 3 points to form a triangle
                natural_feature nf;  // create natural feature structure
                nf.vertex_count = feature.points3d.size();  // number of boundary vertices
                nf.vertices = new glm::vec3[nf.vertex_count];  // allocate vertex storage
                
                // set elevation and color based on feature type
                if (feature.type.find("water") != std::string::npos) {  // water features
                    nf.elevation = WATER_ELEVATION;  // below ground level for water rendering
                    nf.color = glm::vec3(0.2f, 0.4f, 0.8f);  // blue color for water
                } else {  // land features (parks, grass, etc.)
                    nf.elevation = DEFAULT_ELEVATION;  // slightly above ground to prevent z-fighting
                    nf.color = glm::vec3(0.3f, 0.7f, 0.3f);  // green color for vegetation
                }
                
                // copy vertices and apply elevation
                for (int i = 0; i < nf.vertex_count; i++) {
                    nf.vertices[i] = feature.points3d[i];  // copy position
                    nf.vertices[i].y = nf.elevation;  // override y coordinate with feature elevation
                }
                
                // set remaining properties
                nf.centroid = feature.centroid3d;  // center point
                nf.centroid.y = nf.elevation;  // adjust centroid height
                nf.bounding_radius = feature.bounding_radius;  // spatial bounds
                nf.type = feature.type;  // preserve original type string
                
                natural_features.push_back(nf);  // add to feature list
            }
        }
  }

  natural_feature_count = natural_features.size();  // update global count
  std::cout << "Loaded " << natural_feature_count << " natural features\n";
}