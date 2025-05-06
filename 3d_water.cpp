#include "3d_rendering.h"
// water constants - these create realistic water appearance
const float DEFAULT_WAVE_STRENGTH = 0.5f; // how much the water surface moves vertically
const float DEFAULT_WAVE_SPEED = 0.2f; // how fast the waves travel across the surface  
const float DEFAULT_WAVE_FREQUENCY = 1.0f; // how many wave crests appear per unit distance
const float DEFAULT_REFLECTIVITY = 0.7f; // how much the water acts like a mirror
const float DEFAULT_REFRACTION = 1.0f; // how much light bends when entering water
const float DEFAULT_SPECULAR_POWER = 32.0f; // sharpness of bright spots on water 
const glm::vec3 DEFAULT_WATER_COLOR = glm::vec3(0.0f, 0.3f, 0.8f); // deep blue color
const float WATER_ELEVATION_OFFSET = 0.15f; // height above terrain to prevent z-fighting

void Renderer3D::setup_water_shaders() {
    try {
        // compile shaders - these programs run on the gpu to create water effects
        GLuint water_vertex = compile_shader(GL_VERTEX_SHADER, water_vertex_shader);
        GLuint water_fragment = compile_shader(GL_FRAGMENT_SHADER, water_fragment_shader);
        
        // create shader program - links vertex and fragment shaders into a pipeline
        water_shader_program = glCreateProgram();
        glAttachShader(water_shader_program, water_vertex);glAttachShader(water_shader_program, water_fragment);
        glLinkProgram(water_shader_program);
        
        // check for linking errors - ensures gpu pipeline is valid
        GLint success;glGetProgramiv(water_shader_program, GL_LINK_STATUS, &success);
        if (!success) {char info_log[512];glGetProgramInfoLog(water_shader_program, 512, nullptr, info_log);
            std::cerr << "water shader program linking failed: " << info_log << std::endl;
            throw std::runtime_error("water shader program linking failed");}
        
        // cleanup - delete shader objects after linking
        glDeleteShader(water_vertex);glDeleteShader(water_fragment);
        
        // initialize water properties - these control the natural appearance
        water_props = {
            DEFAULT_WAVE_STRENGTH,DEFAULT_WAVE_SPEED,DEFAULT_WAVE_FREQUENCY,
            DEFAULT_REFLECTIVITY,DEFAULT_REFRACTION,DEFAULT_SPECULAR_POWER,DEFAULT_WATER_COLOR};
        
        std::cout << "water shaders initialized successfully" << std::endl;
    } catch (const std::exception& e) {std::cerr << "failed to set up water shaders: " << e.what() << std::endl;throw;}
}

// generate a grid of vertices for water surfaces - creates flat surface that shaders will animate
std::vector<float> Renderer3D::generate_grid(float size, int resolution) {
    std::vector<float> vertices;
    float spacing = size / resolution; // distance between grid points
    
    // generate grid vertices as triangles - each grid cell becomes two triangles
    for (int z = 0; z < resolution; z++) {
        for (int x = 0; x < resolution; x++) {
            // calculate the positions of the corners
            float x0 = x * spacing - size / 2;float z0 = z * spacing - size / 2;
            float x1 = (x + 1) * spacing - size / 2;float z1 = (z + 1) * spacing - size / 2;
            
            // first triangle (top-left, bottom-left, bottom-right)
            vertices.push_back(x0); vertices.push_back(0.0f); vertices.push_back(z0);
            vertices.push_back(x0); vertices.push_back(0.0f); vertices.push_back(z1);
            vertices.push_back(x1); vertices.push_back(0.0f); vertices.push_back(z1);
            
            // second triangle (top-left, bottom-right, top-right)
            vertices.push_back(x0); vertices.push_back(0.0f); vertices.push_back(z0);
            vertices.push_back(x1); vertices.push_back(0.0f); vertices.push_back(z1);
            vertices.push_back(x1); vertices.push_back(0.0f); vertices.push_back(z0);}
    }
    return vertices;
}

// generate texture coordinates for a grid - maps 2d images onto 3d surface
std::vector<float> Renderer3D::generate_grid_tex_coords(int resolution) {
    std::vector<float> tex_coords;
    
    // generate texture coordinates from 0 to 1 for each grid cell
    for (int z = 0; z < resolution; z++) {
        for (int x = 0; x < resolution; x++) {
            float u0 = static_cast<float>(x) / resolution;float v0 = static_cast<float>(z) / resolution;
            float u1 = static_cast<float>(x + 1) / resolution;float v1 = static_cast<float>(z + 1) / resolution;
            
            // add texture coordinates for each triangle vertex
            tex_coords.push_back(u0); tex_coords.push_back(v0);
            tex_coords.push_back(u0); tex_coords.push_back(v1);
            tex_coords.push_back(u1); tex_coords.push_back(v1);
            
            tex_coords.push_back(u0); tex_coords.push_back(v0);
            tex_coords.push_back(u1); tex_coords.push_back(v1);
            tex_coords.push_back(u1); tex_coords.push_back(v0);}
    }
    return tex_coords;
}

// render a water surface with the water shader - this is where the magic happens
void Renderer3D::draw_water_surface(const std::vector<float>& vertices,const std::vector<float>& tex_coords,const water_properties& water_props) {
    if (vertices.empty() || tex_coords.empty()) return;
    
    // enable blending for water transparency - allows seeing through water
    glEnable(GL_BLEND);glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    // use water shader program - activates the water visual effects
    glUseProgram(water_shader_program);
    
    // set transformation matrices - converts 3d world to 2d screen
    glm::mat4 model = glm::mat4(1.0f);
    glm::mat4 view = glm::lookAt(camera_pos, camera_pos + camera_front, camera_up);
    glm::mat4 projection = glm::perspective(glm::radians(45.0f), 1024.0f/768.0f, 0.1f, 10000.0f);
    
    glUniformMatrix4fv(glGetUniformLocation(water_shader_program, "model"), 1, GL_FALSE, glm::value_ptr(model));
    glUniformMatrix4fv(glGetUniformLocation(water_shader_program, "view"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(water_shader_program, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
    
    // set animation parameters - creates moving water effect
    float time = glfwGetTime();glUniform1f(glGetUniformLocation(water_shader_program, "time"), time);
    glUniform3fv(glGetUniformLocation(water_shader_program, "cameraPosition"), 1, glm::value_ptr(camera_pos));
    
    // set wave parameters - controls water surface behavior
    glUniform1f(glGetUniformLocation(water_shader_program, "waveStrength"), water_props.wave_strength);
    glUniform1f(glGetUniformLocation(water_shader_program, "waveSpeed"), water_props.wave_speed);
    glUniform1f(glGetUniformLocation(water_shader_program, "waveFrequency"), water_props.wave_frequency);
    
    // set visual properties - controls water appearance and lighting
    glUniform3fv(glGetUniformLocation(water_shader_program, "waterColor"), 1, glm::value_ptr(water_props.color));
    glUniform3fv(glGetUniformLocation(water_shader_program, "lightPos"), 1, glm::value_ptr(light_position));
    glUniform3fv(glGetUniformLocation(water_shader_program, "viewPos"), 1, glm::value_ptr(camera_pos));
    glUniform1f(glGetUniformLocation(water_shader_program, "reflectivity"), water_props.reflectivity);
    glUniform1f(glGetUniformLocation(water_shader_program, "refractionStrength"), water_props.refraction_strength);
    glUniform1f(glGetUniformLocation(water_shader_program, "specularPower"), water_props.specular_power);
    
    // create and bind opengl resources - prepares vertex data for rendering
    GLuint temp_vao, vbo_position, vbo_tex_coords;
    glGenVertexArrays(1, &temp_vao);glGenBuffers(1, &vbo_position);glGenBuffers(1, &vbo_tex_coords);
    glBindVertexArray(temp_vao);
    
    // upload vertex positions - defines water surface geometry
    glBindBuffer(GL_ARRAY_BUFFER, vbo_position);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    // upload texture coordinates - maps textures to water surface
    glBindBuffer(GL_ARRAY_BUFFER, vbo_tex_coords);
    glBufferData(GL_ARRAY_BUFFER, tex_coords.size() * sizeof(float), tex_coords.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    
    // draw the water surface - triggers the water shader effects
    glDrawArrays(GL_TRIANGLES, 0, vertices.size() / 3);
    
    // cleanup - prevent resource leaks
    glBindVertexArray(0);glDeleteVertexArrays(1, &temp_vao);glDeleteBuffers(1, &vbo_position);glDeleteBuffers(1, &vbo_tex_coords);
    glDisable(GL_BLEND); // restore default blending
}

void Renderer3D::draw_water_feature(const natural_feature& feature) {
    // validate water feature - must have geometry and be a water type
    if (feature.vertex_count < 3) return;
    bool is_water_feature = (feature.type == "lake" || feature.type == "river" || feature.type == "stream");
    if (!is_water_feature) return;
    
    // activate water shader for realistic water rendering
    glUseProgram(water_shader_program);
    
    // set transformation matrices for camera view
    glm::mat4 model = glm::mat4(1.0f);
    glm::mat4 view = glm::lookAt(camera_pos, camera_pos + camera_front, camera_up);
    glm::mat4 projection = glm::perspective(glm::radians(45.0f), 1024.0f/768.0f, 0.1f, 10000.0f);
    
    glUniformMatrix4fv(glGetUniformLocation(water_shader_program, "model"), 1, GL_FALSE, glm::value_ptr(model));
    glUniformMatrix4fv(glGetUniformLocation(water_shader_program, "view"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(water_shader_program, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
    
    // customize water properties based on type - different bodies have different characteristics
    water_properties water_props = this->water_props;
    
    if (feature.type == "lake") {
        water_props.wave_strength = 0.15f;  // gentle waves for still water
        water_props.wave_speed = 0.05f;     // slow movement
        water_props.color = glm::vec3(0.1f, 0.5f, 0.8f); // bright blue 
        water_props.reflectivity = 0.8f;    // more mirror-like
    } else if (feature.type == "river") {
        water_props.wave_strength = 0.25f;  // more waves for flowing water
        water_props.wave_speed = 0.3f;      // faster current
        water_props.color = glm::vec3(0.0f, 0.4f, 0.7f); // darker blue
    } else if (feature.type == "stream") {
        water_props.wave_strength = 0.2f;   // medium waves
        water_props.wave_speed = 0.2f;      // moderate flow
        water_props.color = glm::vec3(0.3f, 0.6f, 0.9f); // lighter blue
    }
    
    // create vertices and texture coordinates from feature geometry
    std::vector<float> vertices;std::vector<float> tex_coords;
    
    // use centroid for fan triangulation - simple method to turn polygon into triangles
    glm::vec3 centroid = feature.centroid;
    centroid.y = feature.elevation + WATER_ELEVATION_OFFSET; // prevent z-fighting with terrain
    
    // create triangles using fan triangulation - each edge forms triangle with center
    for (int i = 0; i < feature.vertex_count; i++) {
        int next_i = (i + 1) % feature.vertex_count;
        
        // get triangle vertices
        glm::vec3 v1 = feature.vertices[i];glm::vec3 v2 = feature.vertices[next_i];
        v1.y = feature.elevation + WATER_ELEVATION_OFFSET;v2.y = feature.elevation + WATER_ELEVATION_OFFSET;
        
        // add triangle vertices
        vertices.push_back(centroid.x); vertices.push_back(centroid.y); vertices.push_back(centroid.z);
        vertices.push_back(v1.x); vertices.push_back(v1.y); vertices.push_back(v1.z);
        vertices.push_back(v2.x); vertices.push_back(v2.y); vertices.push_back(v2.z);
        
        // add texture coordinates based on world position for tiling
        float scale = 0.01f;
        tex_coords.push_back(centroid.x * scale); tex_coords.push_back(centroid.z * scale);
        tex_coords.push_back(v1.x * scale); tex_coords.push_back(v1.z * scale);
        tex_coords.push_back(v2.x * scale); tex_coords.push_back(v2.z * scale);}
    
    // prevent z-fighting - ensures water renders above terrain
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(-1.0f, -1.0f);
    
    // use stencil buffer to prevent water rendering under buildings
    glEnable(GL_STENCIL_TEST);
    glStencilFunc(GL_ALWAYS, 1, 0xFF);
    glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
    glStencilMask(0xFF);
    glClear(GL_STENCIL_BUFFER_BIT);
    
    // draw the water using the prepared vertices and properties
    draw_water_surface(vertices, tex_coords, water_props);
    
    // restore opengl state
    glDisable(GL_STENCIL_TEST);glDisable(GL_POLYGON_OFFSET_FILL);
}