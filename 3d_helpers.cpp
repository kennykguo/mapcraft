#include "3d_rendering.h"
const double kDegreeToRadian = 0.017453292519943295769236907684886;
const double kEarthRadiusInMeters = 6372797.560856;
const double EARTH_RADIUS = 6371000.0f;
const double mid_lat = 0.7627088631478;

// Convert latitude/longitude to 3D coordinates
glm::vec3 latlon_to_3d(const latlon& coord, const latlon& reference) {
  double lat = coord.latitude;
  double lon = coord.longitude;
  double ref_lat = reference.latitude;
  double ref_lon = reference.longitude;

  // Calculate deltas in radians
  double delta_lat = glm::radians(lat - ref_lat);
  double delta_lon = glm::radians(lon - ref_lon);
  double ref_lat_rad = glm::radians(ref_lat);

  // Convert to 3D coordinates (X = east/west, Y = up, Z = north/south)
  float x = static_cast<float>(delta_lon * EARTH_RADIUS * cos(ref_lat_rad));
  float z = static_cast<float>(delta_lat * EARTH_RADIUS);

  return glm::vec3(x, 0.0f, z); // Y is initially set to ground level
}

glm::vec3 Renderer3D::color_to_vec3(const color& c) {
  return glm::vec3(c.red/255.0f, c.green/255.0f, c.blue/255.0f);
}

// Convert latitude/longitude to 3D coordinates
glm::vec3 Renderer3D::latlon_to_3d(const latlon& coord, const latlon& reference) {
  double lat = coord.latitude;
  double lon = coord.longitude;
  double ref_lat = reference.latitude;
  double ref_lon = reference.longitude;

  // Calculate deltas in radians
  double delta_lat = glm::radians(lat - ref_lat);
  double delta_lon = glm::radians(lon - ref_lon);
  double ref_lat_rad = glm::radians(ref_lat);

  // Convert to 3D coordinates (X = east/west, Y = up, Z = north/south)
  float x = static_cast<float>(delta_lon * EARTH_RADIUS * cos(ref_lat_rad));
  float z = static_cast<float>(delta_lat * EARTH_RADIUS);

  return glm::vec3(x, 0.0f, z); // Y is initially set to ground level
}

glm::vec3 color_to_vec3(const color& c) {
  return glm::vec3(c.red/255.0f, c.green/255.0f, c.blue/255.0f);
}

latlon point2d_to_latlon(const point2d &point) {
  double lon = point.x / kEarthRadiusInMeters / cos(mid_lat) / kDegreeToRadian;
  double lat = point.y / kEarthRadiusInMeters / kDegreeToRadian;
  return latlon(lat, lon);
}

