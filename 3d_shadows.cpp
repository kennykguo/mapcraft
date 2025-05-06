#include "3d_rendering.h"

// shadow mapping constants - these define the quality and behavior of shadows
const int SHADOW_WIDTH = 1024;                    // resolution of shadow map (higher = better quality shadows)
const int SHADOW_HEIGHT = 1024;                   // must be power of 2 for optimal GPU performance
const float SHADOW_NEAR_PLANE = 1.0f;             // closest objects that cast shadows
const float SHADOW_FAR_PLANE = 2000.0f;           // furthest objects that cast shadows
const float SHADOW_ORTHO_SIZE = 1000.0f;          // extent of light coverage (scene radius)
const float MIN_WALL_HEIGHT = 5.0f;               // minimum building height for shadow visibility
const glm::vec3 SHADOW_SCENE_CENTER = glm::vec3(0.0f, 0.0f, 0.0f);  // where light looks at

// initialize shadow mapping system - creates framebuffer and shader for shadow generation
void Renderer3D::setup_shadow_mapping() {
    GLuint shadow_vertex = compile_shader(GL_VERTEX_SHADER, shadow_mapping_vertex_shader);  // transforms vertices to light's view space
    GLuint shadow_fragment = compile_shader(GL_FRAGMENT_SHADER, shadow_mapping_fragment_shader);// minimal fragment shader (depth only)
    shadow_shader_program = glCreateProgram();
    glAttachShader(shadow_shader_program, shadow_vertex);
    glAttachShader(shadow_shader_program, shadow_fragment);
    glLinkProgram(shadow_shader_program);
    GLint success;
    glGetProgramiv(shadow_shader_program, GL_LINK_STATUS, &success);
    if (!success) {
        char info_log[512];
        glGetProgramInfoLog(shadow_shader_program, 512, nullptr, info_log);
        std::cerr << "Shadow shader program linking failed: " << info_log << std::endl;
        throw std::runtime_error("Shadow shader program linking failed");
    }
    glDeleteShader(shadow_vertex);
    glDeleteShader(shadow_fragment);
    
    // create shadow framebuffer - off-screen render target for shadow map
    glGenFramebuffers(1, &shadow_map_fbo);  // framebuffer object contains render targets
    
    // create depth texture - stores depth from light's perspective
    glGenTextures(1, &shadow_map_texture);
    glBindTexture(GL_TEXTURE_2D, shadow_map_texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT,    // format for depth storage
                SHADOW_WIDTH, SHADOW_HEIGHT, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
    
    // configure depth texture - how shadow map is sampled
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);  // no interpolation for crisp shadows
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);  // prevents blurry shadow edges
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);  // handles areas outside shadow map
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    float borderColor[] = { 1.0f, 1.0f, 1.0f, 1.0f };     // white border = no shadow outside map
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
    
    // attach depth texture to framebuffer - makes it the render target
    glBindFramebuffer(GL_FRAMEBUFFER, shadow_map_fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, shadow_map_texture, 0);
    glDrawBuffer(GL_NONE);  // no color buffer needed
    glReadBuffer(GL_NONE);  // no reading from color buffer
    
    // verify framebuffer is complete - ensures proper configuration
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "Framebuffer not complete!" << std::endl;
        throw std::runtime_error("Framebuffer not complete");
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0); // return to default framebuffer
    std::cout << "Shadow mapping initialized successfully" << std::endl;
}

// render shadow map - creates depth map from light's perspective
void Renderer3D::render_shadow_map() {
    // early exit if nothing to render shadows for
    if (visible_building_count <= 0 && visible_road_count <= 0) return;
    
    // setup light matrices - defines how light sees the scene
    glm::mat4 light_projection = glm::ortho(-SHADOW_ORTHO_SIZE, SHADOW_ORTHO_SIZE, // orthographic projection for directional light
                                          -SHADOW_ORTHO_SIZE, SHADOW_ORTHO_SIZE, // creates parallel light rays
                                          SHADOW_NEAR_PLANE, SHADOW_FAR_PLANE); // depth range for shadows
                                          
    glm::mat4 light_view = glm::lookAt(light_position, // eye position (light source)
                                      SHADOW_SCENE_CENTER, // look at point (scene center)
                                      glm::vec3(0.0f, 1.0f, 0.0f)); // up vector (world up)
    
    // combine into light space matrix - transforms world to light's view
    glm::mat4 light_space_matrix = light_projection * light_view;
    
    // setup render target for shadow map
    glViewport(0, 0, SHADOW_WIDTH, SHADOW_HEIGHT);  // set viewport to shadow map size
    glBindFramebuffer(GL_FRAMEBUFFER, shadow_map_fbo);  // render to shadow framebuffer
    glClear(GL_DEPTH_BUFFER_BIT);  // clear previous shadow map
    
    // configure rendering state for shadow pass
    glEnable(GL_DEPTH_TEST);  // enable depth testing
    glDisable(GL_CULL_FACE);  // disable face culling for complete shadows
    
    // activate shadow shader
    glUseProgram(shadow_shader_program);
    
    // send light space matrix to shader
    glUniformMatrix4fv(glGetUniformLocation(shadow_shader_program, "lightSpaceMatrix"), 
                      1, GL_FALSE, glm::value_ptr(light_space_matrix));
    
    // render buildings to shadow map
    if (visible_building_count > 0) {
        // get list of visible buildings from gpu
        std::vector<int> h_visible_indices(visible_building_count);
        cuda_check(cudaMemcpy(h_visible_indices.data(), d_visible_building_indices, visible_building_count * sizeof(int), cudaMemcpyDeviceToHost), "Copy visible building indices to host");
        
        // draw each visible building
        for (int i = 0; i < visible_building_count; i++) {
            int building_idx = h_visible_indices[i];
            if (building_idx < 0 || building_idx >= building_count) continue;
            const building& building = buildings[building_idx];
            // generate building wall geometry
            std::vector<float> vertices;
            float wall_height = std::max(MIN_WALL_HEIGHT, building.height);  // ensure visible height
            for (int j = 0; j < building.vertex_count; j++) {
                int next_idx = (j + 1) % building.vertex_count;  // wrap around to first vertex
                // get bottom vertices
                glm::vec3 bottom1 = building.vertices[j];
                glm::vec3 bottom2 = building.vertices[next_idx];
                // create top vertices by extruding upward
                glm::vec3 top1(bottom1.x, wall_height, bottom1.z);
                glm::vec3 top2(bottom2.x, wall_height, bottom2.z);
                // create two triangles for each wall section
                // first triangle: bottom1, bottom2, top2
                vertices.insert(vertices.end(), {bottom1.x, bottom1.y, bottom1.z, 
                                                bottom2.x, bottom2.y, bottom2.z, 
                                                top2.x, top2.y, top2.z});
                // second triangle: bottom1, top2, top1
                vertices.insert(vertices.end(), {bottom1.x, bottom1.y, bottom1.z, 
                                                top2.x, top2.y, top2.z, 
                                                top1.x, top1.y, top1.z});
            }
            
            // set identity model matrix - no object transformation
            glm::mat4 model = glm::mat4(1.0f);
            glUniformMatrix4fv(glGetUniformLocation(shadow_shader_program, "model"), 1, GL_FALSE, glm::value_ptr(model));
            
            // upload and draw vertices
            glBindBuffer(GL_ARRAY_BUFFER, vbo);
            glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_DYNAMIC_DRAW);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
            glEnableVertexAttribArray(0);
            glDrawArrays(GL_TRIANGLES, 0, vertices.size() / 3);

            // add roof to shadow map if building has one
            if (building.has_roof) {
                vertices.clear();
                
                // calculate roof centroid for triangulation
                glm::vec3 roof_center(0.0f, wall_height, 0.0f);
                for (int j = 0; j < building.vertex_count; j++) {
                    roof_center.x += building.vertices[j].x;
                    roof_center.z += building.vertices[j].z;
                }
                roof_center.x /= building.vertex_count;
                roof_center.z /= building.vertex_count;
                
                // create roof triangles from edges to center
                for (int j = 0; j < building.vertex_count; j++) {
                    int next_idx = (j + 1) % building.vertex_count;
                    
                    glm::vec3 v1(building.vertices[j].x, wall_height, building.vertices[j].z);
                    glm::vec3 v2(building.vertices[next_idx].x, wall_height, building.vertices[next_idx].z);
                    
                    // triangle from edge to center
                    vertices.insert(vertices.end(), {v1.x, v1.y, v1.z, 
                                                    v2.x, v2.y, v2.z, 
                                                    roof_center.x, roof_center.y, roof_center.z});
                }
                // draw roof
                glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_DYNAMIC_DRAW);
                glDrawArrays(GL_TRIANGLES, 0, vertices.size() / 3);
            }
        }
    }
    
    // render roads to shadow map
    if (visible_road_count > 0) {
        // get list of visible roads from gpu
        std::vector<int> h_visible_indices(visible_road_count);
        cuda_check(cudaMemcpy(h_visible_indices.data(), d_visible_road_indices, visible_road_count * sizeof(int), cudaMemcpyDeviceToHost), "Copy visible road indices to host");
        
        // draw each visible road
        for (int i = 0; i < visible_road_count; i++) {
            int road_idx = h_visible_indices[i];
            if (road_idx < 0 || road_idx >= road_count) continue;
            const road_segment& road = road_segments[road_idx];
            if (road.vertex_count < 2) continue;  // need at least 2 points for a road
            
            // generate road geometry
            std::vector<float> vertices;
            for (int j = 0; j < road.vertex_count - 1; j++) {
                // get current and next vertices
                glm::vec3 p1 = road.vertices[j];
                glm::vec3 p2 = road.vertices[j + 1];
                
                // calculate road direction and perpendicular
                glm::vec3 dir = glm::normalize(p2 - p1);
                glm::vec3 perp = glm::normalize(glm::cross(dir, glm::vec3(0, 1, 0))); 
                perp *= road.width * 0.5f;  // scale to half road width
                
                // create quad corners
                glm::vec3 v1 = p1 - perp;  // left side start
                glm::vec3 v2 = p1 + perp;  // right side start
                glm::vec3 v3 = p2 - perp;  // left side end
                glm::vec3 v4 = p2 + perp;  // right side end
                
                // apply elevation
                v1.y = v2.y = p1.y + road.elevation;
                v3.y = v4.y = p2.y + road.elevation;
                
                // create triangles for road segment
                vertices.insert(vertices.end(), {v1.x, v1.y, v1.z, v2.x, v2.y, v2.z, v3.x, v3.y, v3.z});
                vertices.insert(vertices.end(), {v2.x, v2.y, v2.z, v4.x, v4.y, v4.z, v3.x, v3.y, v3.z});
            }
            
            // set identity model matrix
            glm::mat4 model = glm::mat4(1.0f);
            glUniformMatrix4fv(glGetUniformLocation(shadow_shader_program, "model"), 1, GL_FALSE, glm::value_ptr(model));
            
            // upload and draw vertices
            glBindBuffer(GL_ARRAY_BUFFER, vbo);
            glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_DYNAMIC_DRAW);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
            glEnableVertexAttribArray(0);
            glDrawArrays(GL_TRIANGLES, 0, vertices.size() / 3);
        }
    }
    
    // restore default rendering state
    glBindFramebuffer(GL_FRAMEBUFFER, 0);  // return to default framebuffer
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    glViewport(0, 0, width, height);  // restore viewport to window size
    
    // store light space matrix for main rendering pass
    this->light_space_matrix = light_space_matrix;
}