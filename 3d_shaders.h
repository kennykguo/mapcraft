// 3d_shaders.h
#ifndef SHADERS_H
#define SHADERS_H

#include "3d_rendering.h"

// declare all shader variables directly in global scope
extern const char* basic_vertex_shader;
extern const char* basic_fragment_shader;
extern const char* phong_vertex_shader;
extern const char* phong_fragment_shader;
extern const char* tess_vertex_shader;
extern const char* tess_control_shader;
extern const char* tess_evaluation_shader;
extern const char* tess_fragment_shader;
extern const char* water_vertex_shader;
extern const char* water_fragment_shader;
extern const char* shadow_mapping_vertex_shader;
extern const char* shadow_mapping_fragment_shader;
extern const char* shadow_receiver_vertex_shader;
extern const char* shadow_receiver_fragment_shader;
extern const char* perlin_noise_function;
extern std::string terrain_vertex_shader_str;
extern const char* terrain_vertex_shader;
extern const char* terrain_fragment_shader;
extern const char* particle_vertex_shader;
extern const char* particle_fragment_shader;
extern const char* phong_textured_vertex_shader;
extern const char* phong_textured_fragment_shader;
extern const char* sky_fragment_shader;
extern const char* sky_vertex_shader;

#endif
