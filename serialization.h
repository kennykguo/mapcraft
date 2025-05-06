#pragma once
#include "3d_rendering.h"

void load_data(latlon ref_latlon);
bool load_street_segments(const std::string& filename, std::vector<street_segment_data>& out_data);
bool load_features(const std::string& filename, std::vector<feature_data>& out_data);
