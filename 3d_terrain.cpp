#include "3d_rendering.h"

// terrain generation constants - these control the landscape appearance and performance
const float TERRAIN_SIZE = 7500.0f;                 // world units extent of terrain from center
const int TERRAIN_RESOLUTION = 220;                 // grid dimensions (higher = more detail)
const float BASE_TERRAIN_HEIGHT = -0.5f;            // base level below ground plane
const float TERRAIN_NOISE_SCALE = 0.02f;            // frequency of height variations
const float TERRAIN_NOISE_HEIGHT = 3.0f;            // amplitude of height variations
const float TERRAIN_BRIGHTNESS_FACTOR = 1.5f;       // color enhancement multiplier
const glm::vec3 BRIGHT_GREEN = glm::vec3(0.5f, 1.0f, 0.4f);  // vibrant grass color

// global terrain data - maintains mesh across frames
std::vector<float> terrain_vertices;                // vertex positions (x,y,z)
std::vector<float> terrain_tex_coords;              // texture coordinates (u,v)
std::vector<unsigned int> terrain_indices;          // triangle connectivity

// opengl resources - persistent gpu objects for terrain
GLuint terrain_vao, terrain_vbo_position, terrain_vbo_tex_coords, terrain_ebo;
glm::vec2 terrain_noise_offset;                     // offset for procedural noise
float terrain_noise_scale;                          // current noise frequency
float terrain_noise_height;                         // current height variation

// initialize terrain shader system - sets up specialized shaders for landscape rendering
void Renderer3D::setup_terrain_shaders() {
    GLuint terrain_vertex = compile_shader(GL_VERTEX_SHADER, terrain_vertex_shader); // applies height map and calculates normals
    GLuint terrain_fragment = compile_shader(GL_FRAGMENT_SHADER, terrain_fragment_shader);// blends textures based on height/slope
    terrain_shader_program = glCreateProgram();
    glAttachShader(terrain_shader_program, terrain_vertex);
    glAttachShader(terrain_shader_program, terrain_fragment);
    glLinkProgram(terrain_shader_program);
    GLint success;
    glGetProgramiv(terrain_shader_program, GL_LINK_STATUS, &success);
    if (!success) {
        char info_log[512];
        glGetProgramInfoLog(terrain_shader_program, 512, nullptr, info_log);
        std::cerr << "Terrain shader program linking failed: " << info_log << std::endl;
        throw std::runtime_error("Terrain shader program linking failed");
    }
    glDeleteShader(terrain_vertex);
    glDeleteShader(terrain_fragment);
    
    // initialize terrain generation parameters
    terrain_noise_scale = TERRAIN_NOISE_SCALE; // frequency of height variations
    terrain_noise_height = TERRAIN_NOISE_HEIGHT; // default height variation
    terrain_noise_offset = glm::vec2(0.0f, 0.0f); // initial noise offset
    
    std::cout << "Terrain shaders initialized successfully" << std::endl;
}

// generate terrain mesh - creates grid-based landscape geometry
void Renderer3D::generate_terrain() {
    // clear existing terrain data for fresh generation
    terrain_vertices.clear();
    terrain_tex_coords.clear();
    terrain_indices.clear();
    
    std::cout << "generating terrain..." << std::endl;
    float grid_spacing = TERRAIN_SIZE / TERRAIN_RESOLUTION;  // distance between vertices
    
    // generate grid vertices - creates height-mapped landscape
    for (int z = 0; z <= TERRAIN_RESOLUTION; z++) {
        for (int x = 0; x <= TERRAIN_RESOLUTION; x++) {
            // calculate world coordinates centered at origin
            float world_x = x * grid_spacing - TERRAIN_SIZE / 2.0f;  // center x coordinate
            float world_z = z * grid_spacing - TERRAIN_SIZE / 2.0f;  // center z coordinate
            
            // add vertex at base height - actual height applied in shader
            terrain_vertices.insert(terrain_vertices.end(), {world_x, BASE_TERRAIN_HEIGHT, world_z});
            
            // calculate texture coordinates - normalized 0 to 1 range
            float tex_u = static_cast<float>(x) / TERRAIN_RESOLUTION;
            float tex_v = static_cast<float>(z) / TERRAIN_RESOLUTION;
            
            terrain_tex_coords.insert(terrain_tex_coords.end(), {tex_u, tex_v});
        }
    }
    
    // generate triangle indices - connects vertices into triangulated surface
    for (int z = 0; z < TERRAIN_RESOLUTION; z++) {
        for (int x = 0; x < TERRAIN_RESOLUTION; x++) {
            // calculate vertex indices for current grid cell
            unsigned int top_left = z * (TERRAIN_RESOLUTION + 1) + x;
            unsigned int top_right = top_left + 1;
            unsigned int bottom_left = (z + 1) * (TERRAIN_RESOLUTION + 1) + x;
            unsigned int bottom_right = bottom_left + 1;
            
            // create two triangles per grid cell - counter-clockwise winding
            terrain_indices.insert(terrain_indices.end(), {top_left, bottom_left, bottom_right});
            terrain_indices.insert(terrain_indices.end(), {top_left, bottom_right, top_right});
        }
    }
    
    // setup gpu resources - prepares buffers for rendering
    if (terrain_vao) {  // cleanup existing resources if they exist
        glDeleteVertexArrays(1, &terrain_vao);
        glDeleteBuffers(1, &terrain_vbo_position);
        glDeleteBuffers(1, &terrain_vbo_tex_coords);
        glDeleteBuffers(1, &terrain_ebo);
    }
    
    // create and configure vertex array object
    glGenVertexArrays(1, &terrain_vao);
    glGenBuffers(1, &terrain_vbo_position);
    glGenBuffers(1, &terrain_vbo_tex_coords);
    glGenBuffers(1, &terrain_ebo);
    
    glBindVertexArray(terrain_vao);  // activate vao to store attribute configuration
    
    // setup position attribute (location 0)
    glBindBuffer(GL_ARRAY_BUFFER, terrain_vbo_position);
    glBufferData(GL_ARRAY_BUFFER, terrain_vertices.size() * sizeof(float), terrain_vertices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    // setup texture coordinate attribute (location 1)
    glBindBuffer(GL_ARRAY_BUFFER, terrain_vbo_tex_coords);
    glBufferData(GL_ARRAY_BUFFER, terrain_tex_coords.size() * sizeof(float), terrain_tex_coords.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    
    // setup element buffer for indexed drawing
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, terrain_ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, terrain_indices.size() * sizeof(unsigned int), terrain_indices.data(), GL_STATIC_DRAW);
    
    glBindVertexArray(0);  // unbind vao
    
    // update terrain generation parameters for pronounced features
    terrain_noise_scale = TERRAIN_NOISE_SCALE;    // larger scale for obvious ridges
    terrain_noise_height = TERRAIN_NOISE_HEIGHT;  // greater height difference
    
    std::cout << "terrain generated with " << terrain_vertices.size() / 3 << " vertices and " 
              << terrain_indices.size() / 3 << " triangles" << std::endl;
}

// render terrain using specialized shader - handles procedural height and texturing
void Renderer3D::render_terrain() {
    if (terrain_vertices.empty() || terrain_indices.empty() || !terrain_shader_program) return;
    
    glUseProgram(terrain_shader_program);
    
    // setup transformation matrices for terrain
    glm::mat4 model = glm::mat4(1.0f);  // identity - terrain at world origin
    glm::mat4 view = glm::lookAt(camera_pos, camera_pos + camera_front, camera_up);
    glm::mat4 projection = glm::perspective(glm::radians(45.0f), 1024.0f/768.0f, 0.1f, 10000.0f);
    
    // send matrices to shader
    glUniformMatrix4fv(glGetUniformLocation(terrain_shader_program, "model"), 1, GL_FALSE, glm::value_ptr(model));
    glUniformMatrix4fv(glGetUniformLocation(terrain_shader_program, "view"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(terrain_shader_program, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
    glUniformMatrix4fv(glGetUniformLocation(terrain_shader_program, "lightSpaceMatrix"), 1, GL_FALSE, glm::value_ptr(light_space_matrix));
    
    // configure procedural noise parameters for height generation
    glUniform1f(glGetUniformLocation(terrain_shader_program, "noiseScale"), terrain_noise_scale);
    glUniform1f(glGetUniformLocation(terrain_shader_program, "noiseHeight"), 0.3f);  // flatter for grass
    glUniform2fv(glGetUniformLocation(terrain_shader_program, "noiseOffset"), 1, glm::value_ptr(terrain_noise_offset));
    
    // setup lighting parameters
    glUniform3fv(glGetUniformLocation(terrain_shader_program, "lightPos"), 1, glm::value_ptr(light_position));
    glUniform3fv(glGetUniformLocation(terrain_shader_program, "viewPos"), 1, glm::value_ptr(camera_pos));
    
    // configure material properties for grass appearance
    glUniform1f(glGetUniformLocation(terrain_shader_program, "ambient"), 0.4f);    // increased ambient for brightness
    glUniform1f(glGetUniformLocation(terrain_shader_program, "diffuse"), 0.6f);    // moderate diffuse lighting
    glUniform1f(glGetUniformLocation(terrain_shader_program, "specular"), 0.0f);   // no specular for grass
    glUniform1f(glGetUniformLocation(terrain_shader_program, "shininess"), 1.0f);  // minimal shininess
    
    // configure texture blending parameters - forces grass appearance
    glUniform1f(glGetUniformLocation(terrain_shader_program, "grassThreshold"), 100.0f);  // everything is grass
    glUniform1f(glGetUniformLocation(terrain_shader_program, "rockThreshold"), 1000.0f);  // rock never appears
    glUniform1f(glGetUniformLocation(terrain_shader_program, "snowThreshold"), 2000.0f);  // snow never appears
    glUniform1f(glGetUniformLocation(terrain_shader_program, "slopeThreshold"), 0.0f);    // disable slope texturing
    
    // bind textures - all slots use grass texture for uniform appearance
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, shadow_map_texture);
    glUniform1i(glGetUniformLocation(terrain_shader_program, "shadowMap"), 0);
    
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, grass_texture);
    glUniform1i(glGetUniformLocation(terrain_shader_program, "grassTexture"), 1);
    
    // bind grass texture to all slots to prevent gl errors
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, grass_texture);
    glUniform1i(glGetUniformLocation(terrain_shader_program, "rockTexture"), 2);
    
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, grass_texture);
    glUniform1i(glGetUniformLocation(terrain_shader_program, "soilTexture"), 3);
    
    // enable polygon offset to prevent z-fighting with roads/buildings
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(1.0f, 1.0f);
    
    // render terrain mesh
    glBindVertexArray(terrain_vao);
    glDrawElements(GL_TRIANGLES, terrain_indices.size(), GL_UNSIGNED_INT, 0);
    
    // restore rendering state
    glDisable(GL_POLYGON_OFFSET_FILL);
    glBindVertexArray(0);
}