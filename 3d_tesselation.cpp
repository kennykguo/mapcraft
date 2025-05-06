#include "3d_rendering.h"

const float MIN_TESS = 1.0f; // minimum subdivision level (less detail)
const float MAX_TESS = 32.0f; // maximum subdivision level (most detail)
const float LOD_FACTOR = 0.001f; // how quickly detail decreases with distance


void Renderer3D::setup_tessellation_shaders() {
    GLuint tess_vertex = compile_shader(GL_VERTEX_SHADER, tess_vertex_shader);
    GLuint tess_control = compile_shader(GL_TESS_CONTROL_SHADER, tess_control_shader);
    GLuint tess_evaluation = compile_shader(GL_TESS_EVALUATION_SHADER, tess_evaluation_shader);
    GLuint tess_fragment = compile_shader(GL_FRAGMENT_SHADER, tess_fragment_shader);
    tess_shader_program = glCreateProgram();
    glAttachShader(tess_shader_program, tess_vertex);glAttachShader(tess_shader_program, tess_control);
    glAttachShader(tess_shader_program, tess_evaluation);glAttachShader(tess_shader_program, tess_fragment);
    glLinkProgram(tess_shader_program);
    GLint success;glGetProgramiv(tess_shader_program, GL_LINK_STATUS, &success);
    if (!success) {char info_log[512];glGetProgramInfoLog(tess_shader_program, 512, nullptr, info_log);
        std::cerr << "tessellation shader program linking failed: " << info_log << std::endl;
        throw std::runtime_error("tessellation shader program linking failed");}
    glDeleteShader(tess_vertex);glDeleteShader(tess_control);glDeleteShader(tess_evaluation);glDeleteShader(tess_fragment);
    
    // set default tessellation level - how much to subdivide geometry
    tessellation_level = 2.0f;
    std::cout << "tessellation shaders initialized successfully" << std::endl;
}


// helper function to create a shader program with tessellation stages - tessellation splits polygons into more detailed geometry at render time
GLuint Renderer3D::create_shader_program_with_tessellation(const char* vertex_source,const char* tess_control_source,const char* tess_eval_source,const char* fragment_source) {
    GLuint vertex = compile_shader(GL_VERTEX_SHADER, vertex_source); // transforms vertex positions
    GLuint tess_control = compile_shader(GL_TESS_CONTROL_SHADER, tess_control_source); // determines subdivision level
    GLuint tess_eval = compile_shader(GL_TESS_EVALUATION_SHADER, tess_eval_source); // generates new vertices
    GLuint fragment = compile_shader(GL_FRAGMENT_SHADER, fragment_source); // colors pixels
    GLuint program = glCreateProgram();
    glAttachShader(program, vertex);glAttachShader(program, tess_control);glAttachShader(program, tess_eval);glAttachShader(program, fragment);
    glLinkProgram(program);
    GLint success;glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {char info_log[512];glGetProgramInfoLog(program, 512, nullptr, info_log);
        std::cerr << "tessellation program linking failed: " << info_log << std::endl;
        throw std::runtime_error("tessellation program linking failed");}
    glDeleteShader(vertex);glDeleteShader(tess_control);glDeleteShader(tess_eval);glDeleteShader(fragment);
    return program;
}

void Renderer3D::draw_with_tessellation(const std::vector<float>& vertices,const std::vector<float>& normals,const glm::vec3& color,const material_properties& material,float tess_level) {
    glUseProgram(tess_shader_program);
    
    // check if tessellation is supported - not all hardware can tesselate
    GLint max_patches;glGetIntegerv(GL_MAX_PATCH_VERTICES, &max_patches);
    if (max_patches < 3) {
        std::cerr << "warning: tessellation not fully supported. using fallback rendering." << std::endl;
        draw_with_phong(vertices, normals, color, material);return;}
    
    // set transform matrices - converts 3d world to 2d screen
    glm::mat4 model = glm::mat4(1.0f);
    glm::mat4 view = glm::lookAt(camera_pos, camera_pos + camera_front, camera_up);
    glm::mat4 projection = glm::perspective(glm::radians(45.0f), 1024.0f/768.0f, 0.1f, 10000.0f);
    
    glUniformMatrix4fv(glGetUniformLocation(tess_shader_program, "model"), 1, GL_FALSE, glm::value_ptr(model));
    glUniformMatrix4fv(glGetUniformLocation(tess_shader_program, "view"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(tess_shader_program, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
    
    // set lighting parameters - controls how light affects the surface
    glUniform3fv(glGetUniformLocation(tess_shader_program, "viewPos"), 1, glm::value_ptr(camera_pos));
    glUniform3fv(glGetUniformLocation(tess_shader_program, "lightPos"), 1, glm::value_ptr(light_position));
    glUniform3fv(glGetUniformLocation(tess_shader_program, "lightColor"), 1, glm::value_ptr(light_color));
    glUniform3fv(glGetUniformLocation(tess_shader_program, "objectColor"), 1, glm::value_ptr(color));
    
    // set material properties - defines surface characteristics
    glUniform1f(glGetUniformLocation(tess_shader_program, "ambient"), material.ambient);
    glUniform1f(glGetUniformLocation(tess_shader_program, "diffuse"), material.diffuse);
    glUniform1f(glGetUniformLocation(tess_shader_program, "specular"), material.specular);
    glUniform1f(glGetUniformLocation(tess_shader_program, "shininess"), material.shininess);
    
    // set distance-based tessellation level - far objects need less detail
    float distance_to_camera = glm::distance(camera_pos, glm::vec3(0.0f));
    float adjusted_tess_level = tess_level * (1.0f / (1.0f + distance_to_camera * LOD_FACTOR));
    adjusted_tess_level = glm::clamp(adjusted_tess_level, MIN_TESS, tess_level);
    
    glUniform1f(glGetUniformLocation(tess_shader_program, "tessellation_level"), adjusted_tess_level);
    
    // create and bind vao and vbos - manages vertex data format
    GLuint temp_vao, vbo_position, vbo_normal;
    glGenVertexArrays(1, &temp_vao);glGenBuffers(1, &vbo_position);glGenBuffers(1, &vbo_normal);
    glBindVertexArray(temp_vao);
    
    // position attribute - vertex locations in 3d space
    glBindBuffer(GL_ARRAY_BUFFER, vbo_position);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    // normal attribute - surface directions for lighting
    glBindBuffer(GL_ARRAY_BUFFER, vbo_normal);
    glBufferData(GL_ARRAY_BUFFER, normals.size() * sizeof(float), normals.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    
    glPatchParameteri(GL_PATCH_VERTICES, 3); // 3 vertices per patch (triangles)
    // draw with patches for tessellation - sends geometry to tessellator
    glDrawArrays(GL_PATCHES, 0, vertices.size() / 3);
    // cleanup - prevent resource leaks
    glBindVertexArray(0);glDeleteVertexArrays(1, &temp_vao);glDeleteBuffers(1, &vbo_position);glDeleteBuffers(1, &vbo_normal);
}
