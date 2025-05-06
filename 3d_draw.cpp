#include "3d_rendering.h"

// constants for rendering properties
const float DEFAULT_AMBIENT = 0.3f;
const float DEFAULT_DIFFUSE = 0.7f;
const float DEFAULT_SPECULAR = 0.2f;
const float DEFAULT_SHININESS = 8.0f;
const float FOV_DEGREES = 45.0f;
const float ASPECT_RATIO = 1024.0f/768.0f;
const float NEAR_PLANE = 0.1f;
const float FAR_PLANE = 10000.0f;
const float TEXTURE_SCALE = 10.0f;
const int ANTENNA_HEIGHT_THRESHOLD = 80;
const int ANTENNA_SEGMENTS = 8;
const float ANTENNA_BASE_RADIUS = 1.0f;
const float ROAD_ELEVATION = 0.5f;
const float MIN_ROAD_WIDTH = 2.0f;
const int WATER_GRID_RESOLUTION = 30;
const float TREE_CHECK_RADIUS = 50.0f;
const float MIN_TREE_DISTANCE = 15.0f;
const int MAX_TREE_ATTEMPTS = 50;
const float ANTENNA_HEIGHT = 0.2;
const int MAX_ROOF_COMPLEXITY = 8;

// copies visible building indices from gpu to host memory
// enables depth testing and polygon offset to prevent z-fighting
// activates phong shader program for rendering
// pre-computes view and projection matrices
// sets lighting parameters and camera position
// binds building textures (concrete and windows)
// configures material properties for realistic appearance
// for each visible building:
    // calculates texture scaling based on building dimensions
    // generates wall vertices, normals, and texture coordinates
    // sets building color based on height
    // uploads geometry data to gpu buffers
    // configures window parameters based on building height
    // renders walls with textured triangles
    // draws roof if building has one
    // disables polygon offset after rendering
void Renderer3D::draw_buildings() {
    // allocate buffer for visible building indices
    if (visible_building_count <= 0) return;
    std::vector<int> h_visible_indices(visible_building_count);
    
    // copy visible indices from gpu
    cuda_check(cudaMemcpy(h_visible_indices.data(), d_visible_building_indices, visible_building_count * sizeof(int), cudaMemcpyDeviceToHost), "Copy visible building indices to host");
    
    // enable depth testing - prevents farther objects from appearing in front of near ones
    // enable polygon offset to prevent z-fighting - nudges one surface slightly forward
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(-0.5f, -0.5f);
    
    // use phong shader for textured buildings
    // shaders are just programs that run on the gpu
    // vertex shader - takes each 3d point to determine where it should be
    // fragment shader - decides what color each pixel should be
    // activates a shader program for rendering - contains instructions for processing vertices and pixels
    glUseProgram(phong_shader_program);
    
    // pre-compute view and projection matrices (used for all buildings)
    glm::mat4 view = glm::lookAt(camera_pos, camera_pos + camera_front, camera_up);
    glm::mat4 projection = glm::perspective(glm::radians(45.0f), 1024.0f/768.0f, 0.1f, 10000.0f);
    
    // set common uniforms
    // glGetUniformLocation - finds location of uniform variable, and used for settings
    // glUniform3fv sends datat o uniform variable to the shader
    // uniforms can be thought of instructions given to shader program - eg. find instruction viewPos in shader program, then execute instruction
    glUniform3fv(glGetUniformLocation(phong_shader_program, "viewPos"), 1, glm::value_ptr(camera_pos));
    glUniform3fv(glGetUniformLocation(phong_shader_program, "lightPos"), 1, glm::value_ptr(light_position));
    glUniform3fv(glGetUniformLocation(phong_shader_program, "lightColor"), 1, glm::value_ptr(light_color));
    
    // bind building textures
    // selects texture unit 0, loading it in
    glActiveTexture(GL_TEXTURE0);
    // bind speciic texture to the unit
    glBindTexture(GL_TEXTURE_2D, concrete_texture);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, window_texture);
    
    // modified material properties for buildings
    glUniform1f(glGetUniformLocation(phong_shader_program, "ambient"), 0.3f);
    glUniform1f(glGetUniformLocation(phong_shader_program, "diffuse"), 0.7f);
    glUniform1f(glGetUniformLocation(phong_shader_program, "specular"), 0.2f);
    glUniform1f(glGetUniformLocation(phong_shader_program, "shininess"), 8.0f);
    
    // draw each visible building
    for (int i = 0; i < visible_building_count; i++) {
        int building_idx = h_visible_indices[i];
        // skip invalid indices
        if (building_idx < 0 || building_idx >= building_count) continue;
        const building& building = buildings[building_idx]; // get the building
        
        // draw building walls with textures
        std::vector<float> wall_vertices;
        std::vector<float> wall_normals;
        std::vector<float> wall_tex_coords;
        
        // make sure height is at least 5.0 meters for visibility
        float wall_height = std::max(5.0f, building.height);
        
        // scale texture based on building size (approx width)
        float building_width = 0;
        for (int j = 0; j < building.vertex_count; j++) {
            int next_idx = (j + 1) % building.vertex_count; // get next vertex index, wrapping around at the end
            float segment_length = glm::distance(
                glm::vec2(building.vertices[j].x, building.vertices[j].z), // current vertex in xz plane
                glm::vec2(building.vertices[next_idx].x, building.vertices[next_idx].z) // next vertex in xz plane
            );
            building_width += segment_length; // accumulate segment length
        }
        building_width /= building.vertex_count; // compute average segment length (approximate width)

        // calculate texture scale based on building width
        float tex_scale_horizontal = building_width / 10.0f; // one texture repeat per 10 meters
        float tex_scale_vertical = wall_height / 5.0f;       // one texture repeat per 5 meters of height
        for (int j = 0; j < building.vertex_count; j++) {
            int next_idx = (j + 1) % building.vertex_count;
            glm::vec3 bottom1 = building.vertices[j];
            glm::vec3 bottom2 = building.vertices[next_idx];
            glm::vec3 top1(bottom1.x, wall_height, bottom1.z);
            glm::vec3 top2(bottom2.x, wall_height, bottom2.z);
            
            // calculate texture scaling
            float segment_length = glm::distance(glm::vec2(bottom1.x, bottom1.z), glm::vec2(bottom2.x, bottom2.z));
            float u_scale = segment_length / TEXTURE_SCALE;
            float v_scale = wall_height / (TEXTURE_SCALE / 2);
            
            // first triangle: bottom1, bottom2, top2
            wall_vertices.insert(wall_vertices.end(), {bottom1.x, bottom1.y, bottom1.z, bottom2.x, bottom2.y, bottom2.z, top2.x, top2.y, top2.z});
            wall_tex_coords.insert(wall_tex_coords.end(), {0.0f, 0.0f, u_scale, 0.0f, u_scale, v_scale});
            
            // second triangle: bottom1, top2, top1
            wall_vertices.insert(wall_vertices.end(), {bottom1.x, bottom1.y, bottom1.z, top2.x, top2.y, top2.z, top1.x, top1.y, top1.z});
            wall_tex_coords.insert(wall_tex_coords.end(), {0.0f, 0.0f, u_scale, v_scale, 0.0f, v_scale});
            
            // calculate normal vector for this wall segment
            glm::vec3 normal = glm::normalize(glm::cross(bottom2 - bottom1, top2 - bottom1));
            for (int k = 0; k < 6; k++) wall_normals.insert(wall_normals.end(), {normal.x, normal.y, normal.z});
        } 
        
        // set building color based on height, but with more stable calculation
        float height_factor = glm::clamp(building.height / 100.0f, 0.0f, 1.0f);
        glm::vec3 building_color = glm::mix(glm::vec3(0.95f, 0.9f, 0.8f), glm::vec3(0.9f, 0.9f, 0.95f), height_factor);
        glUniform3fv(glGetUniformLocation(phong_shader_program, "objectColor"), 1, glm::value_ptr(building_color));

        // position attribute
        // glBindBuffer selects which buffer to work with - containers in the gpu that hold data
        // glBufferData - uploads data from the cpu to the gpu
        // how to interpret data in the buffer - layout of the data
        // activate the vertex attribute
        glBindVertexArray(building_vao);
        glBindBuffer(GL_ARRAY_BUFFER, building_vbo_position);
        glBufferData(GL_ARRAY_BUFFER, wall_vertices.size() * sizeof(float), wall_vertices.data(), GL_DYNAMIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        
        // normal attribute
        glBindBuffer(GL_ARRAY_BUFFER, building_vbo_normal);
        glBufferData(GL_ARRAY_BUFFER, wall_normals.size() * sizeof(float), wall_normals.data(), GL_DYNAMIC_DRAW);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(1);
        
        // texture coordinate attribute
        glBindBuffer(GL_ARRAY_BUFFER, building_vbo_texcoord);
        glBufferData(GL_ARRAY_BUFFER, wall_tex_coords.size() * sizeof(float), wall_tex_coords.data(), GL_DYNAMIC_DRAW);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(2);
        
        // set texture uniforms
        glUniform1i(glGetUniformLocation(phong_shader_program, "wallTexture"), 0);
        glUniform1i(glGetUniformLocation(phong_shader_program, "windowTexture"), 1);
        
        // set use windows flag based on building height
        glUniform1i(glGetUniformLocation(phong_shader_program, "useWindows"), building.height > 10.0f ? 1 : 0);
        
        // set window parameters
        float windowDensity = glm::clamp(building.height / 50.0f, 0.2f, 0.8f); // more windows on taller buildings
        glUniform1f(glGetUniformLocation(phong_shader_program, "windowDensity"), windowDensity);
        
        // draw the building walls
        // takes the data set up, send it through graphics pipeline, draw triangle using every group of 3 vertices, and process al lvertices
        glDrawArrays(GL_TRIANGLES, 0, wall_vertices.size() / 3);
        
        // draw building roof
        if (building.has_roof) {
            // draw the roof with different texture and color
            draw_building_roof(building);
        }
    }
    
    // disable polygon offset
    glDisable(GL_POLYGON_OFFSET_FILL);
}

// renders roof structures for buildings
// uses flat roofs for complex buildings or those under 30m height
// calculates roof geometry with centroid and peak height based on building dimensions
// creates triangular roof mesh connecting building edges to central peak
// generates vertices, normals and texture coordinates for the roof
// sets up gpu buffers for the roof geometry
// applies material properties and textures specific to roofs
// selects roof color based on building height (reddish for short, dark gray for tall)
// draws roof triangles and cleans up gpu resources
// adds antennas to buildings exceeding height threshold
void Renderer3D::draw_building_roof(const building& building) {
    if (!building.has_roof || building.vertex_count < 3) return;
    glUseProgram(phong_shader_program);
    
    // handle complex or low buildings with flat roofs
    if (building.vertex_count > MAX_ROOF_COMPLEXITY || building.height < 30.0f) {
        draw_simple_flat_roof(building);
        return;
    }
    
    // set transformation matrices for roof rendering
    glm::mat4 view = glm::lookAt(camera_pos, camera_pos + camera_front, camera_up);
    glm::mat4 projection = glm::perspective(glm::radians(FOV_DEGREES), ASPECT_RATIO, NEAR_PLANE, FAR_PLANE);
    glm::mat4 model = glm::mat4(1.0f);
    glUniformMatrix4fv(glGetUniformLocation(phong_shader_program, "model"), 1, GL_FALSE, glm::value_ptr(model));
    glUniformMatrix4fv(glGetUniformLocation(phong_shader_program, "view"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(phong_shader_program, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
    
    // set lighting uniforms
    glUniform3fv(glGetUniformLocation(phong_shader_program, "viewPos"), 1, glm::value_ptr(camera_pos));
    glUniform3fv(glGetUniformLocation(phong_shader_program, "lightPos"), 1, glm::value_ptr(light_position));
    glUniform3fv(glGetUniformLocation(phong_shader_program, "lightColor"), 1, glm::value_ptr(light_color));
    
    // calculate roof geometry
    glm::vec3 roof_centroid(0.0f, building.height, 0.0f);
    for (int i = 0; i < building.vertex_count; i++) roof_centroid += glm::vec3(building.vertices[i].x, 0.0f, building.vertices[i].z);
    roof_centroid /= building.vertex_count;
    
    // determine peak height based on building dimensions
    float max_dimension = 0.0f;
    for (int i = 0; i < building.vertex_count; i++) {
        int next_idx = (i + 1) % building.vertex_count;
        float dist = glm::distance(glm::vec2(building.vertices[i].x, building.vertices[i].z), glm::vec2(building.vertices[next_idx].x, building.vertices[next_idx].z));
        max_dimension = std::max(max_dimension, dist);
    }
    float peak_height = building.height + std::max(3.0f, max_dimension * 0.3f);
    
    // create triangular roof mesh
    std::vector<float> vertices, normals, texcoords;
    glm::vec3 ridge_peak = roof_centroid;
    ridge_peak.y = peak_height;
    for (int i = 0; i < building.vertex_count; i++) {
        int next_idx = (i + 1) % building.vertex_count;
        glm::vec3 v1(building.vertices[i].x, building.height, building.vertices[i].z);
        glm::vec3 v2(building.vertices[next_idx].x, building.height, building.vertices[next_idx].z);
        // create triangle from roof edge to peak
        vertices.insert(vertices.end(), {v1.x, v1.y, v1.z, v2.x, v2.y, v2.z, ridge_peak.x, ridge_peak.y, ridge_peak.z});
        // calculate normal and texture coordinates
        glm::vec3 normal = glm::normalize(glm::cross(v2 - v1, ridge_peak - v1));
        for (int j = 0; j < 3; j++) normals.insert(normals.end(), {normal.x, normal.y, normal.z});
        float tex_scale = 0.05f;
        texcoords.insert(texcoords.end(), {v1.x * tex_scale, v1.z * tex_scale, v2.x * tex_scale, v2.z * tex_scale, ridge_peak.x * tex_scale, ridge_peak.z * tex_scale});
    }
    
    // create and bind buffers for roof
    GLuint roof_vao, vbo_position, vbo_normal, vbo_texcoord;
    glGenVertexArrays(1, &roof_vao);
    glGenBuffers(1, &vbo_position);
    glGenBuffers(1, &vbo_normal);
    glGenBuffers(1, &vbo_texcoord);
    
    glBindVertexArray(roof_vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_position);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    glBindBuffer(GL_ARRAY_BUFFER, vbo_normal);
    glBufferData(GL_ARRAY_BUFFER, normals.size() * sizeof(float), normals.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    
    glBindBuffer(GL_ARRAY_BUFFER, vbo_texcoord);
    glBufferData(GL_ARRAY_BUFFER, texcoords.size() * sizeof(float), texcoords.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(2);
    
    // set roof material and color
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, concrete_texture);
    glUniform1i(glGetUniformLocation(phong_shader_program, "mainTexture"), 0);
    glUniform1i(glGetUniformLocation(phong_shader_program, "useTexture"), 1);
    glUniform1i(glGetUniformLocation(phong_shader_program, "useWindows"), 0);
    
    // select roof color by building size
    glm::vec3 roof_color = building.height < 20.0f ? glm::vec3(0.8f, 0.4f, 0.3f) : 
                          building.height < 50.0f ? glm::vec3(0.3f, 0.3f, 0.35f) : 
                          glm::vec3(0.25f, 0.3f, 0.4f);
    glUniform3fv(glGetUniformLocation(phong_shader_program, "objectColor"), 1, glm::value_ptr(roof_color));
    
    // set material properties for roof
    glUniform1f(glGetUniformLocation(phong_shader_program, "ambient"), 0.2f);
    glUniform1f(glGetUniformLocation(phong_shader_program, "diffuse"), 0.7f);
    glUniform1f(glGetUniformLocation(phong_shader_program, "specular"), 0.3f);
    glUniform1f(glGetUniformLocation(phong_shader_program, "shininess"), 32.0f);
    
    // draw roof and cleanup
    glDrawArrays(GL_TRIANGLES, 0, vertices.size() / 3);
    glDeleteVertexArrays(1, &roof_vao);
    glDeleteBuffers(1, &vbo_position);
    glDeleteBuffers(1, &vbo_normal);
    glDeleteBuffers(1, &vbo_texcoord);
    
    // add antenna to tall buildings
    if (building.height >= ANTENNA_HEIGHT_THRESHOLD) draw_building_antenna(building);
}

// draw a simple flat roof for complex buildings
void Renderer3D::draw_simple_flat_roof(const building& building) {
    
    // use phong shader for textured roofs
    glUseProgram(phong_shader_program);
    
    // setup transformation matrices for flat roof rendering
    glm::mat4 view = glm::lookAt(camera_pos, camera_pos + camera_front, camera_up);
    glm::mat4 projection = glm::perspective(glm::radians(FOV_DEGREES), ASPECT_RATIO, NEAR_PLANE, FAR_PLANE);
    glm::mat4 model = glm::mat4(1.0f);
    glUniformMatrix4fv(glGetUniformLocation(phong_shader_program, "model"), 1, GL_FALSE, glm::value_ptr(model));
    glUniformMatrix4fv(glGetUniformLocation(phong_shader_program, "view"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(phong_shader_program, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
    
    // select roof color based on building height
    glm::vec3 roof_color = building.height < 20.0f ? glm::vec3(0.8f, 0.4f, 0.3f) : glm::vec3(0.3f, 0.3f, 0.35f);
    glUniform3fv(glGetUniformLocation(phong_shader_program, "objectColor"), 1, glm::value_ptr(roof_color));
    
    // bind texture for roof
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, concrete_texture);
    glUniform1i(glGetUniformLocation(phong_shader_program, "mainTexture"), 0);
    
    // create flat roof vertices using centroid triangulation
    std::vector<float> vertices, normals, tex_coords;
    glm::vec3 centroid(0.0f, building.height, 0.0f);
    for (int i = 0; i < building.vertex_count; i++) centroid += glm::vec3(building.vertices[i].x, 0.0f, building.vertices[i].z);
    centroid /= building.vertex_count;
    // build roof from centroid to edges using triangles
    for (int i = 0; i < building.vertex_count; i++) {
        int next_i = (i + 1) % building.vertex_count;
        glm::vec3 v1(building.vertices[i].x, building.height, building.vertices[i].z);
        glm::vec3 v2(building.vertices[next_i].x, building.height, building.vertices[next_i].z);
        // add triangle vertices
        vertices.insert(vertices.end(), {centroid.x, centroid.y, centroid.z, v1.x, v1.y, v1.z, v2.x, v2.y, v2.z});
        // add upward-facing normals
        for (int j = 0; j < 3; j++) normals.insert(normals.end(), {0.0f, 1.0f, 0.0f});
        // add texture coordinates
        float tex_scale = 0.05f;
        tex_coords.insert(tex_coords.end(), {centroid.x * tex_scale, centroid.z * tex_scale, v1.x * tex_scale, v1.z * tex_scale, v2.x * tex_scale, v2.z * tex_scale});
    }
    
    // create and bind buffers for flat roof
    GLuint roof_vao, vbo_position, vbo_normal, vbo_texcoord;
    glGenVertexArrays(1, &roof_vao); // creates vertex array object - container for attribute configurations
    glGenBuffers(1, &vbo_position);  // creates buffer for position data
    glGenBuffers(1, &vbo_normal);    // creates buffer for normal data
    glGenBuffers(1, &vbo_texcoord);  // creates buffer for texture coordinate data
    
    glBindVertexArray(roof_vao); // activates the VAO to store attribute settings
    
    // configure position attribute
    glBindBuffer(GL_ARRAY_BUFFER, vbo_position);  // select the position buffer
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);  // upload position data
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);  // describe data layout (3 floats per vertex)
    glEnableVertexAttribArray(0);  // enable position attribute
    
    // configure normal attribute
    glBindBuffer(GL_ARRAY_BUFFER, vbo_normal);
    glBufferData(GL_ARRAY_BUFFER, normals.size() * sizeof(float), normals.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    
    // configure texture coordinate attribute
    glBindBuffer(GL_ARRAY_BUFFER, vbo_texcoord);
    glBufferData(GL_ARRAY_BUFFER, tex_coords.size() * sizeof(float), tex_coords.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);  // texture coords use 2 floats (u,v)
    glEnableVertexAttribArray(2);
    
    // configure shader settings
    glUniform1i(glGetUniformLocation(phong_shader_program, "useTexture"), 1);   // enable texture usage
    glUniform1i(glGetUniformLocation(phong_shader_program, "useWindows"), 0);   // disable window overlay
    
    // set material properties for roof
    glUniform1f(glGetUniformLocation(phong_shader_program, "ambient"), 0.2f);
    glUniform1f(glGetUniformLocation(phong_shader_program, "diffuse"), 0.7f);
    glUniform1f(glGetUniformLocation(phong_shader_program, "specular"), 0.3f);
    glUniform1f(glGetUniformLocation(phong_shader_program, "shininess"), 32.0f);
    
    // draw the flat roof
    glDrawArrays(GL_TRIANGLES, 0, vertices.size() / 3);  // draw all triangles (3 vertices each)
    
    // cleanup - delete temporary objects
    glDeleteVertexArrays(1, &roof_vao);
    glDeleteBuffers(1, &vbo_position);
    glDeleteBuffers(1, &vbo_normal);
    glDeleteBuffers(1, &vbo_texcoord);
}

// draw antenna on top of tall buildings
void Renderer3D::draw_building_antenna(const building& building) {
    // check building height threshold
    if (building.height < ANTENNA_HEIGHT_THRESHOLD) return;
    
    // calculate roof center position
    glm::vec3 roof_center(0.0f, building.height, 0.0f);
    for (int i = 0; i < building.vertex_count; i++) roof_center += glm::vec3(building.vertices[i].x, 0.0f, building.vertices[i].z);
    roof_center /= building.vertex_count;
    
    // determine antenna dimensions
    float antenna_height = building.height * ANTENNA_HEIGHT;  // 15% of building height
    float antenna_radius = ANTENNA_BASE_RADIUS;
    
    // create antenna geometry (cylinder + cone)
    std::vector<float> vertices;
    int segments = ANTENNA_SEGMENTS;
    glm::vec3 base_pos = roof_center;
    
    // create cylinder part (80% of antenna height)
    float cylinder_height = antenna_height * 0.8f;
    for (int i = 0; i < segments; i++) {
        float angle1 = glm::radians(static_cast<float>(i) / segments * 360.0f);
        float angle2 = glm::radians(static_cast<float>(i + 1) / segments * 360.0f);
        // bottom vertices
        glm::vec3 bottom1 = base_pos + glm::vec3(cos(angle1) * antenna_radius, 0.0f, sin(angle1) * antenna_radius);
        glm::vec3 bottom2 = base_pos + glm::vec3(cos(angle2) * antenna_radius, 0.0f, sin(angle2) * antenna_radius);
        // top vertices
        glm::vec3 top1 = bottom1 + glm::vec3(0.0f, cylinder_height, 0.0f);
        glm::vec3 top2 = bottom2 + glm::vec3(0.0f, cylinder_height, 0.0f);
        // add cylinder side triangles
        vertices.insert(vertices.end(), {bottom1.x, bottom1.y, bottom1.z, bottom2.x, bottom2.y, bottom2.z, top2.x, top2.y, top2.z});
        vertices.insert(vertices.end(), {bottom1.x, bottom1.y, bottom1.z, top2.x, top2.y, top2.z, top1.x, top1.y, top1.z});
    }
    
    // create cone part (20% of antenna height)
    float cone_height = antenna_height * 0.2f;
    glm::vec3 cone_base = base_pos + glm::vec3(0.0f, cylinder_height, 0.0f);
    glm::vec3 cone_tip = cone_base + glm::vec3(0.0f, cone_height, 0.0f);
    for (int i = 0; i < segments; i++) {
        float angle1 = glm::radians(static_cast<float>(i) / segments * 360.0f);
        float angle2 = glm::radians(static_cast<float>(i + 1) / segments * 360.0f);
        // base vertices
        glm::vec3 base1 = cone_base + glm::vec3(cos(angle1) * antenna_radius, 0.0f, sin(angle1) * antenna_radius);
        glm::vec3 base2 = cone_base + glm::vec3(cos(angle2) * antenna_radius, 0.0f, sin(angle2) * antenna_radius);
        // add cone triangles
        vertices.insert(vertices.end(), {base1.x, base1.y, base1.z, base2.x, base2.y, base2.z, cone_tip.x, cone_tip.y, cone_tip.z});
    }
    
    // calculate normals for lighting
    std::vector<float> normals = calculate_normals(vertices);
    // set antenna properties
    glm::vec3 antenna_color(0.3f, 0.3f, 0.3f);
    material_properties antenna_material = {0.2f, 0.5f, 0.7f, 32.0f};
    // draw antenna with phong lighting
    draw_with_phong(vertices, normals, antenna_color, antenna_material);
    // add blinking light at the top
    std::vector<float> light_vertices;
    glm::vec3 light_pos = cone_tip + glm::vec3(0.0f, 0.5f, 0.0f);
    float light_radius = 0.5f;
    
    // create sphere for light (simplified)
    int stacks = 4, sectors = 8;
    for (int i = 0; i < stacks; i++) {
        float stack_angle1 = glm::radians(static_cast<float>(i) / stacks * 180.0f);
        float stack_angle2 = glm::radians(static_cast<float>(i + 1) / stacks * 180.0f);
        float y1 = cos(stack_angle1);
        float y2 = cos(stack_angle2);
        float r1 = sin(stack_angle1);
        float r2 = sin(stack_angle2);
        for (int j = 0; j < sectors; j++) {
            float sector_angle1 = glm::radians(static_cast<float>(j) / sectors * 360.0f);
            float sector_angle2 = glm::radians(static_cast<float>(j + 1) / sectors * 360.0f);
            // calculate sphere vertex positions
            float x1 = r1 * cos(sector_angle1), z1 = r1 * sin(sector_angle1);
            float x2 = r1 * cos(sector_angle2), z2 = r1 * sin(sector_angle2);
            float x3 = r2 * cos(sector_angle1), z3 = r2 * sin(sector_angle1);
            float x4 = r2 * cos(sector_angle2), z4 = r2 * sin(sector_angle2);
            // add sphere triangles (skip poles)
            if (i != 0) light_vertices.insert(light_vertices.end(), {
                light_pos.x + x1 * light_radius, light_pos.y + y1 * light_radius, light_pos.z + z1 * light_radius,
                light_pos.x + x3 * light_radius, light_pos.y + y2 * light_radius, light_pos.z + z3 * light_radius,
                light_pos.x + x2 * light_radius, light_pos.y + y1 * light_radius, light_pos.z + z2 * light_radius});
            if (i != stacks - 1) light_vertices.insert(light_vertices.end(), {
                light_pos.x + x3 * light_radius, light_pos.y + y2 * light_radius, light_pos.z + z3 * light_radius,
                light_pos.x + x4 * light_radius, light_pos.y + y2 * light_radius, light_pos.z + z4 * light_radius,
                light_pos.x + x2 * light_radius, light_pos.y + y1 * light_radius, light_pos.z + z2 * light_radius});
        }
    }
    
    // calculate light sphere normals
    std::vector<float> light_normals = calculate_normals(light_vertices);
    
    // make light blink
    float time = glfwGetTime();  // gets current time since start
    bool light_on = ((int)(time * 2) % 2) == 0;  // blink every 0.5 seconds
    
    // set light color based on state
    glm::vec3 light_color = light_on ? glm::vec3(1.0f, 0.0f, 0.0f) : glm::vec3(0.5f, 0.0f, 0.0f);
    material_properties light_material = {0.8f, 1.0f, 1.0f, 64.0f};
    
    // draw blinking light
    draw_with_phong(light_vertices, light_normals, light_color, light_material);
}

// draw road segments
void Renderer3D::draw_roads() {
    if (visible_road_count <= 0) return;
    
    // get visible road indices from GPU
    std::vector<int> h_visible_indices(visible_road_count);
    cuda_check(cudaMemcpy(h_visible_indices.data(), d_visible_road_indices, visible_road_count * sizeof(int), cudaMemcpyDeviceToHost), "Copy visible road indices to host");
    
    // setup shader and matrices
    glUseProgram(phong_shader_program);
    glm::mat4 view = glm::lookAt(camera_pos, camera_pos + camera_front, camera_up);
    glm::mat4 projection = glm::perspective(glm::radians(FOV_DEGREES), ASPECT_RATIO, NEAR_PLANE, FAR_PLANE);
    
    // set uniforms
    glUniform3fv(glGetUniformLocation(phong_shader_program, "viewPos"), 1, glm::value_ptr(camera_pos));
    glUniform3fv(glGetUniformLocation(phong_shader_program, "lightPos"), 1, glm::value_ptr(light_position));
    glUniform3fv(glGetUniformLocation(phong_shader_program, "lightColor"), 1, glm::value_ptr(light_color));
    
    // bind road texture
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, road_texture);
    glUniform1i(glGetUniformLocation(phong_shader_program, "mainTexture"), 0);
    
    // set road material properties
    glUniform1f(glGetUniformLocation(phong_shader_program, "ambient"), 0.3f);
    glUniform1f(glGetUniformLocation(phong_shader_program, "diffuse"), 0.7f);
    glUniform1f(glGetUniformLocation(phong_shader_program, "specular"), 0.1f);
    glUniform1f(glGetUniformLocation(phong_shader_program, "shininess"), 4.0f);
    
    // enable polygon offset for road layering
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(-1.0f, -1.0f);
    
    // set transformation matrices
    glm::mat4 model = glm::mat4(1.0f);
    glUniformMatrix4fv(glGetUniformLocation(phong_shader_program, "model"), 1, GL_FALSE, glm::value_ptr(model));
    glUniformMatrix4fv(glGetUniformLocation(phong_shader_program, "view"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(phong_shader_program, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
    
    // draw each visible road segment
    for (int i = 0; i < visible_road_count; i++) {
        int road_idx = h_visible_indices[i];
        if (road_idx < 0 || road_idx >= road_count) continue;
        
        const road_segment& road = road_segments[road_idx];
        if (road.vertex_count < 2) continue;  // need at least 2 vertices for a segment
        
        // create road geometry
        std::vector<float> vertices, normals, tex_coords;
        
        // build quads for each road segment
        for (int j = 0; j < road.vertex_count - 1; j++) {
            glm::vec3 p1 = road.vertices[j];
            glm::vec3 p2 = road.vertices[j + 1];
            // calculate perpendicular direction for road width
            glm::vec3 dir = glm::normalize(p2 - p1);
            glm::vec3 perp = glm::normalize(glm::cross(dir, glm::vec3(0, 1, 0)));
            // ensure minimum road width
            float road_width = std::max(MIN_ROAD_WIDTH, road.width);
            perp *= road_width * 0.5f;
            // calculate quad corners
            glm::vec3 v1 = p1 - perp, v2 = p1 + perp;
            glm::vec3 v3 = p2 - perp, v4 = p2 + perp;
            // set elevation
            v1.y = v2.y = v3.y = v4.y = ROAD_ELEVATION;
            // calculate texture scaling
            float segment_length = glm::distance(glm::vec2(p1.x, p1.z), glm::vec2(p2.x, p2.z));
            float tex_scale = segment_length / TEXTURE_SCALE;
            // add first triangle
            vertices.insert(vertices.end(), {v1.x, v1.y, v1.z, v2.x, v2.y, v2.z, v3.x, v3.y, v3.z});
            tex_coords.insert(tex_coords.end(), {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, tex_scale});
            // add second triangle
            vertices.insert(vertices.end(), {v2.x, v2.y, v2.z, v4.x, v4.y, v4.z, v3.x, v3.y, v3.z});
            tex_coords.insert(tex_coords.end(), {1.0f, 0.0f, 1.0f, tex_scale, 0.0f, tex_scale});
            // add upward-facing normals
            for (int k = 0; k < 6; k++) normals.insert(normals.end(), {0.0f, 1.0f, 0.0f});
        }
        if (vertices.empty()) continue;
        
        // set road color by type
        glm::vec3 road_color;
        switch (road.road_type) {
            case 0: road_color = glm::vec3(0.6f, 0.6f, 0.6f); break;   // highway
            case 1: road_color = glm::vec3(0.65f, 0.65f, 0.65f); break; // primary
            case 2: road_color = glm::vec3(0.7f, 0.7f, 0.7f); break;   // secondary
            case 3: road_color = glm::vec3(0.75f, 0.75f, 0.75f); break; // residential
            case 4: road_color = glm::vec3(0.8f, 0.75f, 0.7f); break;  // footpath
            default: road_color = glm::vec3(0.7f, 0.7f, 0.7f); break;  // default
        }
        glUniform3fv(glGetUniformLocation(phong_shader_program, "objectColor"), 1, glm::value_ptr(road_color));
        
        // create and bind buffers for this road
        GLuint road_vao, vbo_position, vbo_normal, vbo_texcoord;
        glGenVertexArrays(1, &road_vao);
        glGenBuffers(1, &vbo_position);
        glGenBuffers(1, &vbo_normal);
        glGenBuffers(1, &vbo_texcoord);
        
        glBindVertexArray(road_vao);
        
        // setup attributes
        glBindBuffer(GL_ARRAY_BUFFER, vbo_position);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_DYNAMIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        
        glBindBuffer(GL_ARRAY_BUFFER, vbo_normal);
        glBufferData(GL_ARRAY_BUFFER, normals.size() * sizeof(float), normals.data(), GL_DYNAMIC_DRAW);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(1);
        
        glBindBuffer(GL_ARRAY_BUFFER, vbo_texcoord);
        glBufferData(GL_ARRAY_BUFFER, tex_coords.size() * sizeof(float), tex_coords.data(), GL_DYNAMIC_DRAW);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(2);
        
        // draw and cleanup
        glDrawArrays(GL_TRIANGLES, 0, vertices.size() / 3);
        glDeleteVertexArrays(1, &road_vao);
        glDeleteBuffers(1, &vbo_position);
        glDeleteBuffers(1, &vbo_normal);
        glDeleteBuffers(1, &vbo_texcoord);
    }
    
    glDisable(GL_POLYGON_OFFSET_FILL);
}

// draw natural features
// need to fix - not all water bodies are closed ways
void Renderer3D::draw_natural_features() {
    if (visible_natural_feature_count <= 0 || !d_visible_natural_feature_indices) return;
    
    try {
        // get visible feature indices
        std::vector<int> h_visible_indices(visible_natural_feature_count);
        cuda_check(cudaMemcpy(h_visible_indices.data(), d_visible_natural_feature_indices, visible_natural_feature_count * sizeof(int), cudaMemcpyDeviceToHost), "Copy visible natural feature indices to host");
        
        glEnable(GL_DEPTH_TEST);
        
        // draw non-water features first
        for (int i = 0; i < visible_natural_feature_count; i++) {
            int feature_idx = h_visible_indices[i];
            if (feature_idx < 0 || feature_idx >= natural_feature_count) continue;
            
            const natural_feature& feature = natural_features[feature_idx];
            if (feature.vertex_count < 3) continue;
            
            // skip water features in this pass
            bool is_water = (feature.type == "lake" || feature.type == "river" || feature.type == "stream");
            if (is_water) continue;
            
            // create hill geometry for parks/greenspace
            std::vector<float> vertices;
            glm::vec3 centroid(0.0f);
            for (int j = 0; j < feature.vertex_count; j++) centroid += feature.vertices[j];
            centroid /= feature.vertex_count;
            
            // calculate average distance for hill creation
            float avg_dist = 0.0f;
            for (int j = 0; j < feature.vertex_count; j++) {
                avg_dist += glm::distance(glm::vec2(feature.vertices[j].x, feature.vertices[j].z), glm::vec2(centroid.x, centroid.z));
            }
            avg_dist /= feature.vertex_count;
            
            // create hill triangles
            for (int j = 1; j < feature.vertex_count - 1; j++) {
                float base_height = 0.1f;
                
                // calculate height factors
                float dist1 = glm::distance(glm::vec2(feature.vertices[0].x, feature.vertices[0].z), glm::vec2(centroid.x, centroid.z));
                float dist2 = glm::distance(glm::vec2(feature.vertices[j].x, feature.vertices[j].z), glm::vec2(centroid.x, centroid.z));
                float dist3 = glm::distance(glm::vec2(feature.vertices[j+1].x, feature.vertices[j+1].z), glm::vec2(centroid.x, centroid.z));
                
                float height_factor1 = glm::max(0.0f, 1.0f - (dist1 / avg_dist));
                float height_factor2 = glm::max(0.0f, 1.0f - (dist2 / avg_dist));
                float height_factor3 = glm::max(0.0f, 1.0f - (dist3 / avg_dist));
                
                float hill_height = 4.0f;
                
                // add triangle vertices with elevation
                vertices.insert(vertices.end(), {
                    feature.vertices[0].x, base_height + height_factor1 * hill_height, feature.vertices[0].z,
                    feature.vertices[j].x, base_height + height_factor2 * hill_height, feature.vertices[j].z,
                    feature.vertices[j+1].x, base_height + height_factor3 * hill_height, feature.vertices[j+1].z
                });
            }
            
            // render greenspace with texture
            if (feature.type == "greenspace" || feature.type == "park") {
                render_darker_greenspace(feature);
            } else {
                // basic rendering for other features
                glUseProgram(shader_program);
                glEnable(GL_POLYGON_OFFSET_FILL);
                glPolygonOffset(-1.5f, -1.5f);
                
                glm::mat4 model = glm::mat4(1.0f);
                glm::mat4 view = glm::lookAt(camera_pos, camera_pos + camera_front, camera_up);
                glm::mat4 projection = glm::perspective(glm::radians(FOV_DEGREES), ASPECT_RATIO, NEAR_PLANE, FAR_PLANE);
                
                glUniformMatrix4fv(glGetUniformLocation(shader_program, "model"), 1, GL_FALSE, glm::value_ptr(model));
                glUniformMatrix4fv(glGetUniformLocation(shader_program, "view"), 1, GL_FALSE, glm::value_ptr(view));
                glUniformMatrix4fv(glGetUniformLocation(shader_program, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
                
                // enhance color for better visibility
                glm::vec3 enhanced_color = feature.color * 1.5f;
                glUniform3f(glGetUniformLocation(shader_program, "color"), enhanced_color.x, enhanced_color.y, enhanced_color.z);
                
                // upload vertices to GPU
                glBindVertexArray(vao);
                glBindBuffer(GL_ARRAY_BUFFER, vbo);
                glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);
                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
                glEnableVertexAttribArray(0);
                
                // draw triangles
                glDrawArrays(GL_TRIANGLES, 0, vertices.size() / 3);
                glDisable(GL_POLYGON_OFFSET_FILL);
            }
        }
        
        // now process water features
        for (int i = 0; i < visible_natural_feature_count; i++) {
            int feature_idx = h_visible_indices[i];
            if (feature_idx < 0 || feature_idx >= natural_feature_count) continue;
            const natural_feature& feature = natural_features[feature_idx];
            draw_water_feature(feature);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error in draw_natural_features: " << e.what() << std::endl;
    }
}
    // count existing trees in area
    int trees_in_greenspace = 0;
    float check_radius = TREE_CHECK_RADIUS;
    
    for (const auto& tree : trees) {
        float dist = glm::distance(glm::vec2(tree.position.x, tree.position.z), glm::vec2(feature.centroid.x, feature.centroid.z));
        if (dist < check_radius) trees_in_greenspace++;
    }
    
    // return if enough trees already present
    if (trees_in_greenspace >= 3) return;
    
    // prepare random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // calculate area of greenspace
    float area = 0;
    for (int i = 0; i < feature.vertex_count; i++) {
        int next_i = (i + 1) % feature.vertex_count;
        glm::vec3 v1 = feature.vertices[i];
        glm::vec3 v2 = feature.vertices[next_i];
        glm::vec3 cross_product = glm::cross(v1 - feature.centroid, v2 - feature.centroid);
        area += 0.5f * glm::length(cross_product);
    }
    
    // determine number of trees to add
    int trees_to_add = std::min(3, std::max(1, 3 - trees_in_greenspace));
    float min_tree_distance = MIN_TREE_DISTANCE;
    
    // setup distributions for random tree properties
    std::uniform_real_distribution<float> scale_dist(0.7f, 1.3f);
    std::uniform_real_distribution<float> rot_dist(0.0f, 360.0f);
    
    // collect existing tree positions for collision checking
    std::vector<glm::vec2> existing_positions;
    for (const auto& tree : trees) existing_positions.push_back(glm::vec2(tree.position.x, tree.position.z));
    
    // attempt to place trees
    int attempts = 0;
    for (int i = 0; i < trees_to_add && attempts < MAX_TREE_ATTEMPTS; i++) {
        bool valid_position = false;
        glm::vec3 position;
        
        // try to find valid position
        for (int attempt = 0; attempt < MAX_TREE_ATTEMPTS/trees_to_add && !valid_position; attempt++) {
            attempts++;
            
            // interpolate position along greenspace edge
            int idx1 = rand() % feature.vertex_count;
            int idx2 = (idx1 + 1) % feature.vertex_count;
            float t = std::uniform_real_distribution<float>(0.2f, 0.8f)(gen);
            
            position = feature.vertices[idx1] * (1.0f - t) + feature.vertices[idx2] * t;
            
            // move inward from edge
            glm::vec3 to_centroid = glm::normalize(feature.centroid - position);
            float inward_dist = std::uniform_real_distribution<float>(5.0f, 20.0f)(gen);
            position += to_centroid * inward_dist;
            
            // check if position is within bounds
            float dist_to_centroid = glm::distance(glm::vec2(position.x, position.z), glm::vec2(feature.centroid.x, feature.centroid.z));
            if (dist_to_centroid > 0.9f * check_radius) continue;
            
            // check distance from other trees
            valid_position = true;
            for (const auto& existing_pos : existing_positions) {
                float dist = glm::distance(glm::vec2(position.x, position.z), existing_pos);
                if (dist < min_tree_distance) {
                    valid_position = false;
                    break;
                }
            }
        }
        
        // place tree if valid position found
        if (valid_position) {
            position.y = feature.elevation + 0.1f;
            
            tree_mesh tree;
            tree.position = position;
            tree.scale = scale_dist(gen);
            tree.rotation = rot_dist(gen);
            tree.type = 0;  // pine tree type
            
            trees.push_back(tree);
            existing_positions.push_back(glm::vec2(position.x, position.z));
        }
    }
}