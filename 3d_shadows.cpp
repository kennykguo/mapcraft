#include "3d_rendering.h"

// shadow mapping constants
const int SHADOW_WIDTH = 1024;                    // resolution of shadow map
const int SHADOW_HEIGHT = 1024;                   // must be power of 2 for performance
const float SHADOW_NEAR_PLANE = 1.0f;             // closest objects for shadows
const float SHADOW_FAR_PLANE = 2000.0f;           // furthest objects for shadows
const float SHADOW_ORTHO_SIZE = 1000.0f;          // shadow coverage extent
const float MIN_WALL_HEIGHT = 5.0f;               // min building height for shadows
const glm::vec3 SHADOW_SCENE_CENTER = glm::vec3(0.0f, 0.0f, 0.0f);  // shadow focus point

// setup_shadow_mapping - initializes shadow mapping system
// - creates framebuffer and depth texture for shadow rendering
// - compiles specialized shaders for shadow pass
// - creates a separate render target for the light's view
// - enables realistic shadows from directional light
void Renderer3D::setup_shadow_mapping() {
    // compile shadow shaders
    // - vertex shader transforms positions to light space
    // - fragment shader only needs to output depth values
    GLuint shadow_vertex = compile_shader(GL_VERTEX_SHADER, shadow_mapping_vertex_shader);
    GLuint shadow_fragment = compile_shader(GL_FRAGMENT_SHADER, shadow_mapping_fragment_shader);
    
    // create and link shadow program
    // - links vertex and fragment shaders
    // - program will render the scene from light's perspective
    shadow_shader_program = glCreateProgram();
    glAttachShader(shadow_shader_program, shadow_vertex);
    glAttachShader(shadow_shader_program, shadow_fragment);
    glLinkProgram(shadow_shader_program);
    
    // verify program linking success
    GLint success;
    glGetProgramiv(shadow_shader_program, GL_LINK_STATUS, &success);
    if (!success) {
        char info_log[512];
        glGetProgramInfoLog(shadow_shader_program, 512, nullptr, info_log);
        std::cerr << "shadow shader program linking failed: " << info_log << std::endl;
        throw std::runtime_error("shadow shader program linking failed");
    }
    
    // cleanup individual shaders
    glDeleteShader(shadow_vertex);
    glDeleteShader(shadow_fragment);
    
    // create framebuffer for shadow rendering
    // - framebuffer is a container for render targets
    // - allows rendering to texture instead of screen
    glGenFramebuffers(1, &shadow_map_fbo);
    
    // create depth texture
    // - stores depth values from light's perspective
    // - will be used to determine if fragments are in shadow
    glGenTextures(1, &shadow_map_texture);
    glBindTexture(GL_TEXTURE_2D, shadow_map_texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT,
                SHADOW_WIDTH, SHADOW_HEIGHT, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
    
    // configure depth texture parameters
    // - GL_NEAREST filter for crisp shadows
    // - clamp to border with white for areas outside shadow map
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    float borderColor[] = { 1.0f, 1.0f, 1.0f, 1.0f };
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
    
    // configure framebuffer
    // - attach depth texture to framebuffer
    // - only need depth component for shadow mapping
    // - disable color buffers (not needed for depth-only rendering)
    glBindFramebuffer(GL_FRAMEBUFFER, shadow_map_fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, shadow_map_texture, 0);
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);
    
    // verify framebuffer completeness
    // - ensures proper configuration before use
    // - incomplete framebuffers can cause rendering issues
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "framebuffer not complete!" << std::endl;
        throw std::runtime_error("framebuffer not complete");
    }
    
    // restore default framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    std::cout << "shadow mapping initialized successfully" << std::endl;
}

// render_shadow_map - creates depth map from light's perspective
// - generates shadows by rendering scene from light's viewpoint
// - captures depth information only (no color)
// - creates data for comparing fragment depths during main rendering
// - allows detecting if fragments are in shadow or lit
void Renderer3D::render_shadow_map() {
    // early exit if nothing to render shadows for
    if (visible_building_count <= 0 && visible_road_count <= 0) return;
    
    // setup light matrices for shadow mapping
    // - orthographic projection for directional light
    // - captures large scene area consistently
    // - transforms world coordinates to light space
    glm::mat4 light_projection = glm::ortho(-SHADOW_ORTHO_SIZE, SHADOW_ORTHO_SIZE,
                                          -SHADOW_ORTHO_SIZE, SHADOW_ORTHO_SIZE,
                                          SHADOW_NEAR_PLANE, SHADOW_FAR_PLANE);
                                          
    // create light view matrix
    // - positions virtual camera at light source position
    // - looks towards scene center
    // - transforms vertices to light's perspective
    glm::mat4 light_view = glm::lookAt(light_position,
                                      SHADOW_SCENE_CENTER,
                                      glm::vec3(0.0f, 1.0f, 0.0f));
    
    // combine into single transformation
    // - light_space_matrix transforms world coordinates directly to light clip space
    // - used for shadow mapping calculations in fragment shader
    glm::mat4 light_space_matrix = light_projection * light_view;
    
    // configure opengl for shadow pass
    // - set viewport to shadow map dimensions
    // - bind shadow framebuffer as render target
    // - clear previous depth information
    glViewport(0, 0, SHADOW_WIDTH, SHADOW_HEIGHT);
    glBindFramebuffer(GL_FRAMEBUFFER, shadow_map_fbo);
    glClear(GL_DEPTH_BUFFER_BIT);
    
    // set opengl state for shadow rendering
    // - enable depth testing to capture correct depths
    // - disable face culling for complete shadows
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    
    // activate shadow shader for depth-only rendering
    glUseProgram(shadow_shader_program);
    
    // send light transformation to shader
    // - allows shader to convert world positions to light space
    // - critical for depth comparison in shadow test
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
            
            // generate building geometry for shadow pass
            // - creates simplified wall geometry for shadow casting
            // - uses minimum height threshold for visual consistency
            std::vector<float> vertices;
            float wall_height = std::max(MIN_WALL_HEIGHT, building.height);
            
            // create wall geometry
            // - constructs triangles for each wall segment
            // - forms complete silhouette for shadow casting
            for (int j = 0; j < building.vertex_count; j++) {
                int next_idx = (j + 1) % building.vertex_count;
                
                // get vertices for wall corners
                glm::vec3 bottom1 = building.vertices[j];
                glm::vec3 bottom2 = building.vertices[next_idx];
                glm::vec3 top1(bottom1.x, wall_height, bottom1.z);
                glm::vec3 top2(bottom2.x, wall_height, bottom2.z);
                
                // create two triangles for each wall
                // first triangle: bottom1, bottom2, top2
                vertices.insert(vertices.end(), {bottom1.x, bottom1.y, bottom1.z, 
                                                bottom2.x, bottom2.y, bottom2.z, 
                                                top2.x, top2.y, top2.z});
                // second triangle: bottom1, top2, top1
                vertices.insert(vertices.end(), {bottom1.x, bottom1.y, bottom1.z, 
                                                top2.x, top2.y, top2.z, 
                                                top1.x, top1.y, top1.z});
            }
            
            // set identity model matrix
            // - no additional transformation needed for buildings
            // - building vertices already in world space
            glm::mat4 model = glm::mat4(1.0f);
            glUniformMatrix4fv(glGetUniformLocation(shadow_shader_program, "model"), 1, GL_FALSE, glm::value_ptr(model));
            
            // upload vertices and render wall geometry
            // - uploads wall vertices to gpu
            // - configures attribute pointer for vertex positions
            // - draws triangles from uploaded data
            glBindBuffer(GL_ARRAY_BUFFER, vbo);
            glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_DYNAMIC_DRAW);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
            glEnableVertexAttribArray(0);
            glDrawArrays(GL_TRIANGLES, 0, vertices.size() / 3);

            // add roof to shadow map if building has one
            if (building.has_roof) {
                vertices.clear();
                
                // calculate roof center for triangulation
                // - computes average of all vertex positions
                // - used as center point for fan triangulation
                glm::vec3 roof_center(0.0f, wall_height, 0.0f);
                for (int j = 0; j < building.vertex_count; j++) {
                    roof_center.x += building.vertices[j].x;
                    roof_center.z += building.vertices[j].z;
                }
                roof_center.x /= building.vertex_count;
                roof_center.z /= building.vertex_count;
                
                // create roof triangles from edge to center
                // - forms triangular fan from perimeter to center
                // - creates complete roof coverage for shadows
                for (int j = 0; j < building.vertex_count; j++) {
                    int next_idx = (j + 1) % building.vertex_count;
                    
                    glm::vec3 v1(building.vertices[j].x, wall_height, building.vertices[j].z);
                    glm::vec3 v2(building.vertices[next_idx].x, wall_height, building.vertices[next_idx].z);
                    
                    // triangle from edge to center
                    vertices.insert(vertices.end(), {v1.x, v1.y, v1.z, 
                                                    v2.x, v2.y, v2.z, 
                                                    roof_center.x, roof_center.y, roof_center.z});
                }
                
                // upload and draw roof geometry
                glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_DYNAMIC_DRAW);
                glDrawArrays(GL_TRIANGLES, 0, vertices.size() / 3);
            }
        }
    }
    
    // render roads to shadow map if enabled
    // - roads normally don't cast significant shadows
    // - included for completeness in shadow mapping
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
            
            // generate road geometry for shadow map
            // - creates road segment quads for shadow casting
            // - each segment is a strip between consecutive points
            std::vector<float> vertices;
            for (int j = 0; j < road.vertex_count - 1; j++) {
                // get current and next vertices
                glm::vec3 p1 = road.vertices[j];
                glm::vec3 p2 = road.vertices[j + 1];
                
                // calculate road direction and perpendicular
                // - finds perpendicular vector to create width
                // - scales based on road width parameter
                glm::vec3 dir = glm::normalize(p2 - p1);
                glm::vec3 perp = glm::normalize(glm::cross(dir, glm::vec3(0, 1, 0))); 
                perp *= road.width * 0.5f;  // scale to half road width
                
                // create quad corners
                // - generates 4 corners of road segment
                // - slightly elevated to prevent z-fighting
                glm::vec3 v1 = p1 - perp;  // left side start
                glm::vec3 v2 = p1 + perp;  // right side start
                glm::vec3 v3 = p2 - perp;  // left side end
                glm::vec3 v4 = p2 + perp;  // right side end
                
                // adjust height for shadow mapping
                v1.y += ROAD_ELEVATION;
                v2.y += ROAD_ELEVATION;
                v3.y += ROAD_ELEVATION;
                v4.y += ROAD_ELEVATION;
                
                // create two triangles for quad
                // - first triangle: v1, v2, v3
                // - second triangle: v2, v4, v3
                vertices.insert(vertices.end(), {v1.x, v1.y, v1.z, 
                                                v2.x, v2.y, v2.z, 
                                                v3.x, v3.y, v3.z});
                                                
                vertices.insert(vertices.end(), {v2.x, v2.y, v2.z, 
                                                v4.x, v4.y, v4.z, 
                                                v3.x, v3.y, v3.z});
            }
            
            // set identity model matrix for road
            glm::mat4 model = glm::mat4(1.0f);
            glUniformMatrix4fv(glGetUniformLocation(shadow_shader_program, "model"), 1, GL_FALSE, glm::value_ptr(model));
            
            // upload vertices and render road geometry
            glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_DYNAMIC_DRAW);
            glDrawArrays(GL_TRIANGLES, 0, vertices.size() / 3);
        }
    }
    
    // restore default framebuffer and viewport
    // - returns rendering to main screen
    // - resets viewport to window dimensions
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, 1024, 768);
}