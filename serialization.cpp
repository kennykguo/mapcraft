#include <fstream>
#include <sstream>
#include <zlib.h>
#include <cstring>
#include "serialization.h"

// Compression helper functions
std::string compress_string(const std::string& str) {
    z_stream zs;
    memset(&zs, 0, sizeof(zs));

    if (deflateInit(&zs, Z_BEST_COMPRESSION) != Z_OK) {
        throw std::runtime_error("deflateInit failed");
    }

    zs.next_in = (Bytef*)str.data();
    zs.avail_in = str.size();

    int ret;
    char outbuffer[32768];
    std::string outstring;

    do {
        zs.next_out = reinterpret_cast<Bytef*>(outbuffer);
        zs.avail_out = sizeof(outbuffer);

        ret = deflate(&zs, Z_FINISH);

        if (outstring.size() < zs.total_out) {
            outstring.append(outbuffer, zs.total_out - outstring.size());
        }
    } while (ret == Z_OK);

    deflateEnd(&zs);

    if (ret != Z_STREAM_END) {
        throw std::runtime_error("compression failed");
    }

    return outstring;
}

std::string decompress_string(const std::string& str) {
    z_stream zs;
    memset(&zs, 0, sizeof(zs));

    if (inflateInit(&zs) != Z_OK) {
        throw std::runtime_error("inflateInit failed");
    }

    zs.next_in = (Bytef*)str.data();
    zs.avail_in = str.size();

    int ret;
    char outbuffer[32768];
    std::string outstring;

    do {
        zs.next_out = reinterpret_cast<Bytef*>(outbuffer);
        zs.avail_out = sizeof(outbuffer);

        ret = inflate(&zs, 0);

        if (outstring.size() < zs.total_out) {
            outstring.append(outbuffer, zs.total_out - outstring.size());
        }
    } while (ret == Z_OK);

    inflateEnd(&zs);

    if (ret != Z_STREAM_END) {
        throw std::runtime_error("decompression failed");
    }

    return outstring;
}

template <typename T>
void write_pod(std::ostream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
void read_pod(std::istream& in, T& value) {
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
}

void write_string(std::ostream& out, const std::string& str) {
    uint32_t length = str.size();
    write_pod(out, length);
    out.write(str.data(), length);
}

void read_string(std::istream& in, std::string& str) {
    uint32_t length;
    read_pod(in, length);
    str.resize(length);
    in.read(&str[0], length);
}

template <typename T>
void write_vector(std::ostream& out, const std::vector<T>& vec) {
    uint32_t size = vec.size();
    write_pod(out, size);
    out.write(reinterpret_cast<const char*>(vec.data()), size * sizeof(T));
}

template <typename T>
void read_vector(std::istream& in, std::vector<T>& vec) {
    uint32_t size;
    read_pod(in, size);
    vec.resize(size);
    in.read(reinterpret_cast<char*>(vec.data()), size * sizeof(T));
}

bool load_street_segments(const std::string& filename, std::vector<street_segment_data>& out_data) {
  std::ifstream in(filename, std::ios::binary);
  if (!in) return false;
  
  std::string compressed((std::istreambuf_iterator<char>(in)), 
                       std::istreambuf_iterator<char>());
  
  std::string decompressed;
  try {
      decompressed = decompress_string(compressed);
  } catch (...) {
      return false;
  }
  
  std::istringstream iss(decompressed, std::ios::binary);
  
  uint32_t count;
  read_pod(iss, count);
  out_data.resize(count);
  
  for (auto& item : out_data) {
      read_pod(iss, item.index);
      read_pod(iss, item.osm_id);
      read_pod(iss, item.one_way);
      read_pod(iss, item.speed_limit);
      read_pod(iss, item.street_id);
      read_string(iss, item.street_name);
      read_string(iss, item.highway_type);
      read_pod(iss, item.highway_type_index);
      read_vector(iss, item.points);
      read_pod(iss, item.bounding_box);
      read_pod(iss, item.centroid);
      read_pod(iss, item.width);
      read_pod(iss, item.height);
      read_pod(iss, item.bounding_radius);
      read_vector(iss, item.points3d);
      read_pod(iss, item.centroid3d);
  }
  
  return iss.good();
}

// change this name later since theres function overloading
bool load_features(const std::string& filename, std::vector<feature_data>& out_data) {
  std::ifstream in(filename, std::ios::binary);
  if (!in) return false;
  
  std::string compressed((std::istreambuf_iterator<char>(in)), 
                       std::istreambuf_iterator<char>());
  
  std::string decompressed;
  try {
      decompressed = decompress_string(compressed);
  } catch (...) {
      return false;
  }
  
  std::istringstream iss(decompressed, std::ios::binary);
  
  uint32_t count;
  read_pod(iss, count);
  out_data.resize(count);
  
  for (auto& item : out_data) {
      read_pod(iss, item.index);
      read_string(iss, item.type);
      read_string(iss, item.name);
      read_pod(iss, item.osm_id);
      read_vector(iss, item.points);
      read_pod(iss, item.bounding_box);
      read_pod(iss, item.is_closed);
      read_pod(iss, item.centroid);
      read_pod(iss, item.area);
      read_pod(iss, item.height);
      read_pod(iss, item.bounding_radius);
      read_vector(iss, item.points3d);
      read_pod(iss, item.centroid3d);
      read_pod(iss, item.is_skyscraper);
      read_pod(iss, item.roof_type);
      read_pod(iss, item.has_roof);
      read_pod(iss, item.rgba);
  }
  
  return iss.good();
}

void load_serialized_data(const latlon& ref_latlon) {
    // Load serialized data
    if (!load_street_segments("street_segments.bin", graphics_street_segment_data)) {
        std::cerr << "failed to load street segments" << std::endl;
        return;
    }
    if (!load_features("features.bin", graphics_feature_data)) {
        std::cerr << "failed to load features" << std::endl;
        return;
    }

    // IMPORTANT: Set global reference point
    reference_point = ref_latlon;
    
    std::cout << "Reference point in load_serialized_data: lat=" << reference_point.latitude 
              << ", lon=" << reference_point.longitude << std::endl;
    
    // Debug the coordinate conversion
    glm::vec3 test_point = latlon_to_3d(reference_point, reference_point);
    std::cout << "Reference point 3D (should be ~0,0,0): [" 
              << test_point.x << ", " << test_point.y << ", " << test_point.z << "]" << std::endl;

    std::cout << "Recalculating 3D coordinates with new reference point..." << std::endl;
    
    // ---------------------------------------------------------------------------
    // Process features - NO RANDOM MODIFICATIONS
    // ---------------------------------------------------------------------------
    std::cout << "Processing " << graphics_feature_data.size() << " features..." << std::endl;

    std::cout << "Converting features to 3D coordinates..." << std::endl;

    for (auto& feature : graphics_feature_data) {
        // Clear any existing 3D data
        feature.points3d.clear();
        feature.points3d.reserve(feature.points.size());
        
        // Convert 2D points to 3D for all features
        for (const auto& point : feature.points) {
            // Convert the 2D projected coordinates back to latitude/longitude
            latlon point_latlon = point2d_to_latlon(point);
            
            // Convert to 3D
            glm::vec3 point3d = latlon_to_3d(point_latlon, reference_point);
            
            // Apply elevation WITHOUT MODIFICATIONS
            if (feature.type == "building" || feature.type == "BUILDING") {
                // Buildings start at ground level - use original height data
                point3d.y = DEFAULT_ELEVATION;
            } else if (feature.type == "water") {
                point3d.y = WATER_ELEVATION;
            } else {
                // Other features use their original height or default
                point3d.y = (feature.height > 0) ? feature.height : DEFAULT_ELEVATION;
            }
            
            feature.points3d.push_back(point3d);
        }
        
        // Calculate 3D centroid
        if (!feature.points3d.empty()) {
            glm::vec3 centroid3d(0.0f, 0.0f, 0.0f);
            for (const auto& point : feature.points3d) {
                centroid3d += point;
            }
            centroid3d /= static_cast<float>(feature.points3d.size());
            
            // Set appropriate ground level for centroid
            if (feature.type == "building" || feature.type == "BUILDING") {
                centroid3d.y = DEFAULT_ELEVATION;
            } else if (feature.type == "water") {
                centroid3d.y = WATER_ELEVATION;
            }
            
            feature.centroid3d.x = centroid3d.x;
            feature.centroid3d.y = centroid3d.y;
            feature.centroid3d.z = centroid3d.z;
            
            // Calculate bounding radius
            float max_dist_sq = 0.0f;
            for (const auto& point : feature.points3d) {
                glm::vec3 point_glm(point.x, point.y, point.z);
                glm::vec3 diff = point_glm - centroid3d;
                float dist_sq = glm::dot(diff, diff);
                max_dist_sq = std::max(max_dist_sq, dist_sq);
            }
            
            feature.bounding_radius = std::sqrt(max_dist_sq);
            
            // For buildings, ensure bounding radius includes height (use original height)
            if (feature.type == "building" || feature.type == "BUILDING") {
                float height_component = feature.height / 2.0f;
                feature.bounding_radius = std::sqrt(max_dist_sq + height_component * height_component);
            }
        }
    }

    // ---------------------------------------------------------------------------
    // Process street segments with correct coordinate conversion
    // ---------------------------------------------------------------------------
    std::cout << "Processing " << graphics_street_segment_data.size() << " street segments..." << std::endl;
    
    for (auto& segment : graphics_street_segment_data) {
        // Determine street width based on highway type - NO MODIFICATIONS TO ORIGINAL DATA
        segment.width = 8.0f; // DEFAULT_STREET_WIDTH
        
        // Adjust width based on highway type
        std::string highway_type = segment.highway_type;
        std::transform(highway_type.begin(), highway_type.end(), highway_type.begin(), ::tolower);
        
        if (highway_type == "motorway" || highway_type == "trunk" || 
            highway_type == "motorway_link" || highway_type == "trunk_link") {
            segment.width = 16.0f;  // Wide highways
        } else if (highway_type == "primary" || highway_type == "primary_link") {
            segment.width = 12.0f;  // Primary roads
        } else if (highway_type == "secondary" || highway_type == "secondary_link") {
            segment.width = 10.0f;  // Secondary roads
        } else if (highway_type == "tertiary" || highway_type == "tertiary_link") {
            segment.width = 8.0f;   // Tertiary roads
        } else if (highway_type == "residential" || highway_type == "unclassified") {
            segment.width = 6.0f;   // Residential streets
        } else if (highway_type == "service" || highway_type == "track") {
            segment.width = 4.0f;   // Service roads
        } else if (highway_type == "footway" || highway_type == "path" || 
                  highway_type == "cycleway" || highway_type == "pedestrian") {
            segment.width = 2.0f;   // Footpaths and cycleways
        }
        
        // Convert 2D points to 3D
        segment.points3d.clear();
        segment.points3d.reserve(segment.points.size());
        
        for (const auto& point : segment.points) {
            latlon point_latlon = point2d_to_latlon(point);
            glm::vec3 point3d = latlon_to_3d(point_latlon, reference_point);
            point3d.y = segment.height;
            segment.points3d.push_back(point3d);
        }
        
        // Calculate 3D centroid
        if (!segment.points3d.empty()) {
            glm::vec3 centroid3d(0.0f, 0.0f, 0.0f);
            for (const auto& point : segment.points3d) {
                centroid3d += point;
            }
            centroid3d /= static_cast<float>(segment.points3d.size());
            
            segment.centroid3d.x = centroid3d.x;
            segment.centroid3d.y = centroid3d.y;
            segment.centroid3d.z = centroid3d.z;
            
            // Calculate bounding radius
            float max_dist_sq = 0.0f;
            for (const auto& point : segment.points3d) {
                glm::vec3 point_glm(point.x, point.y, point.z);
                glm::vec3 diff = point_glm - centroid3d;
                float dist_sq = glm::dot(diff, diff);
                max_dist_sq = std::max(max_dist_sq, dist_sq);
            }
            
            segment.bounding_radius = std::sqrt(max_dist_sq);
            segment.bounding_radius += segment.width / 2.0f; // Add half width for better accuracy
        }
    }
    
    std::cout << "Finished loading and converting coordinates:" << std::endl;
    std::cout << "  " << graphics_feature_data.size() << " features loaded" << std::endl;
    std::cout << "  " << graphics_street_segment_data.size() << " street segments loaded" << std::endl;
    
    // Debug output
    if (!graphics_street_segment_data.empty()) {
        const auto& seg = graphics_street_segment_data[0];
        std::cout << "\n=== FIRST STREET SEGMENT (ORIGINAL DATA) ===\n";
        if (!seg.points3d.empty()) {
            std::cout << "First 3D Point: " 
                      << seg.points3d[0].x << ", " 
                      << seg.points3d[0].y << ", "
                      << seg.points3d[0].z << "\n";
        }
        std::cout << "3D Centroid: " 
                  << seg.centroid3d.x << ", "
                  << seg.centroid3d.y << ", "
                  << seg.centroid3d.z << "\n";
    }
    
    if (!graphics_feature_data.empty()) {
        // Find first building
        for (const auto& feature : graphics_feature_data) {
            if (feature.type == "building" || feature.type == "BUILDING") {
                std::cout << "\n=== FIRST BUILDING (ORIGINAL DATA) ===\n";
                std::cout << "Type: " << feature.type << "\n";
                if (!feature.points3d.empty()) {
                    std::cout << "First 3D Point: " 
                            << feature.points3d[0].x << ", " 
                            << feature.points3d[0].y << ", "
                            << feature.points3d[0].z << "\n";
                }
                std::cout << "3D Centroid: " 
                        << feature.centroid3d.x << ", "
                        << feature.centroid3d.y << ", "
                        << feature.centroid3d.z << "\n";
                std::cout << "Original Height: " << feature.height << "m\n";
                break;
            }
        }
    }
}