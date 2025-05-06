#include "3d_rendering.h"

// phong lighting constants - these control how light interacts with surfaces
const glm::vec3 DEFAULT_LIGHT_POSITION = glm::vec3(500.0f, 1000.0f, 500.0f);  // sun position in sky
const glm::vec3 BRIGHT_LIGHT_COLOR = glm::vec3(1.0f, 1.0f, 1.0f);            // pure white light
const float BRIGHTNESS_MULTIPLIER = 1.5f;                                     // factor to brighten colors
const float AMBIENT_BOOST = 2.0f;                                            // multiplier for ambient lighting
const float DIFFUSE_REDUCTION = 0.7f;                                        // factor to reduce diffuse intensity
const float SPECULAR_REDUCTION = 0.5f;                                       // factor to reduce specular highlights

// setup phong lighting system - prepares shaders for realistic 3d illumination
void Renderer3D::setup_phong_shaders() {
    // compile shaders - convert shader source to gpu executable code
    GLuint phong_vertex = compile_shader(GL_VERTEX_SHADER, phong_textured_vertex_shader);      // handles vertex transformation and lighting calculations per vertex
    GLuint phong_fragment = compile_shader(GL_FRAGMENT_SHADER, phong_textured_fragment_shader);// determines final pixel color with lighting and textures
    
    // create shader program - links vertex and fragment shaders into complete pipeline
    phong_shader_program = glCreateProgram();                    // creates empty program container
    glAttachShader(phong_shader_program, phong_vertex);          // adds vertex stage
    glAttachShader(phong_shader_program, phong_fragment);        // adds fragment stage
    glLinkProgram(phong_shader_program);                         // combines stages into executable program
    
    // verify shader linking success
    GLint success;
    glGetProgramiv(phong_shader_program, GL_LINK_STATUS, &success);
    if (!success) {
        char info_log[512];
        glGetProgramInfoLog(phong_shader_program, 512, nullptr, info_log);
        std::cerr << "phong shader program linking failed: " << info_log << std::endl;
        throw std::runtime_error("Phong shader program linking failed");
    }
    glDeleteShader(phong_vertex);
    glDeleteShader(phong_fragment);
    light_position = DEFAULT_LIGHT_POSITION;  // places light source above and to the side of scene
    light_color = BRIGHT_LIGHT_COLOR;         // white light ensures no color tinting
    std::cout << "phong lighting shaders initialized with enhanced texturing capabilities" << std::endl;
}

// render 3d geometry with phong lighting model - creates realistic shading and highlights
void Renderer3D::draw_with_phong(const std::vector<float>& vertices,    // 3d positions defining geometry 
                                const std::vector<float>& normals,      // surface normal vectors for lighting calculations
                                const glm::vec3& color,                 // base color of the object
                                const material_properties& material) {  // physical properties affecting light interaction
    
    glUseProgram(phong_shader_program);  // activate phong shader for this rendering
    
    // setup camera and projection transforms - positions geometry in 3d space
    glm::mat4 model = glm::mat4(1.0f);   // identity matrix (no object transformation)
    glm::mat4 view = glm::lookAt(camera_pos, camera_pos + camera_front, camera_up);  // camera position/orientation
    glm::mat4 projection = glm::perspective(glm::radians(45.0f), 1024.0f/768.0f, 0.1f, 10000.0f);  // 3d to 2d projection
    
    // send transformation matrices to shader - tells gpu how to transform vertices
    glUniformMatrix4fv(glGetUniformLocation(phong_shader_program, "model"), 1, GL_FALSE, glm::value_ptr(model));
    glUniformMatrix4fv(glGetUniformLocation(phong_shader_program, "view"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(phong_shader_program, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
    
    // configure lighting parameters - defines how light affects this object
    glUniform3fv(glGetUniformLocation(phong_shader_program, "viewPos"), 1, glm::value_ptr(camera_pos));      // camera position for specular calculations
    glUniform3fv(glGetUniformLocation(phong_shader_program, "lightPos"), 1, glm::value_ptr(light_position)); // light source position
    glUniform3fv(glGetUniformLocation(phong_shader_program, "lightColor"), 1, glm::value_ptr(light_color));  // light color/intensity
    
    // enhance color brightness - makes objects more visible in the scene
    glm::vec3 brightened_color = color * BRIGHTNESS_MULTIPLIER;  // multiply by 1.5 to increase visibility
    brightened_color = glm::clamp(brightened_color, glm::vec3(0.0f), glm::vec3(1.0f));  // prevent values exceeding valid range
    glUniform3fv(glGetUniformLocation(phong_shader_program, "objectColor"), 1, glm::value_ptr(brightened_color));
    
    // adjust lighting components for more even illumination
    // ambient: base visibility without direct light - increased for brighter appearance
    float enhanced_ambient = std::min(0.8f, material.ambient * AMBIENT_BOOST);
    // diffuse: light scattering across surface - reduced to soften shadows
    float reduced_diffuse = material.diffuse * DIFFUSE_REDUCTION;
    // specular: shiny highlights - reduced to prevent overblown reflections
    float reduced_specular = material.specular * SPECULAR_REDUCTION;
    
    // send material properties to shader - controls how surface responds to light
    glUniform1f(glGetUniformLocation(phong_shader_program, "ambient"), enhanced_ambient);
    glUniform1f(glGetUniformLocation(phong_shader_program, "diffuse"), reduced_diffuse);
    glUniform1f(glGetUniformLocation(phong_shader_program, "specular"), reduced_specular);
    glUniform1f(glGetUniformLocation(phong_shader_program, "shininess"), material.shininess);  // controls specular highlight size
    
    // setup vertex data for rendering - temporary vao and vbos for this draw call
    GLuint temp_vao, vbo_position, vbo_normal;
    glGenVertexArrays(1, &temp_vao);    // create vertex array object to store attribute config
    glGenBuffers(1, &vbo_position);     // create buffer for position data
    glGenBuffers(1, &vbo_normal);       // create buffer for normal data
    
    glBindVertexArray(temp_vao);  // activate vao to capture following state
    
    // configure position attribute (location 0)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_position);  // select position buffer
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_DYNAMIC_DRAW);  // upload position data
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);  // describe data layout (3 floats per vertex)
    glEnableVertexAttribArray(0);  // enable this attribute for shader access
    
    // configure normal attribute (location 1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_normal);  // select normal buffer
    glBufferData(GL_ARRAY_BUFFER, normals.size() * sizeof(float), normals.data(), GL_DYNAMIC_DRAW);  // upload normal data
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);  // describe data layout (3 floats per normal)
    glEnableVertexAttribArray(1);  // enable this attribute for shader access
    
    // execute draw call - renders all triangles at once
    glDrawArrays(GL_TRIANGLES, 0, vertices.size() / 3);  // each triangle uses 3 vertices (9 floats)
    
    // cleanup resources - delete temporary opengl objects
    glBindVertexArray(0);  // unbind vao
    glDeleteVertexArrays(1, &temp_vao);
    glDeleteBuffers(1, &vbo_position);
    glDeleteBuffers(1, &vbo_normal);
}

// compute surface normals for triangle mesh - essential for lighting calculations
std::vector<float> Renderer3D::calculate_normals(const std::vector<float>& vertices) {
    std::vector<float> normals(vertices.size(), 0.0f);  // allocate space matching vertex data
    
    // process each triangle (vertices are stored sequentially as x,y,z triplets)
    for (size_t i = 0; i < vertices.size(); i += 9) {  // step by 9 floats (3 vertices * 3 coordinates)
        if (i + 8 >= vertices.size()) break;  // ensure we have complete triangle data
        
        // extract triangle vertices from flat array
        glm::vec3 v1(vertices[i], vertices[i+1], vertices[i+2]);        // first vertex
        glm::vec3 v2(vertices[i+3], vertices[i+4], vertices[i+5]);      // second vertex
        glm::vec3 v3(vertices[i+6], vertices[i+7], vertices[i+8]);      // third vertex
        
        // calculate two edge vectors of the triangle
        glm::vec3 edge1 = v2 - v1;  // vector from v1 to v2
        glm::vec3 edge2 = v3 - v1;  // vector from v1 to v3
        
        // compute normal vector using cross product - gives perpendicular to triangle plane
        glm::vec3 normal = glm::normalize(glm::cross(edge1, edge2));  // normalize ensures unit length
        
        // assign same normal to all vertices of this triangle
        for (int j = 0; j < 3; j++) {  // loop through the 3 vertices
            normals[i + j*3] = normal.x;      // x component of normal
            normals[i + j*3 + 1] = normal.y;  // y component of normal
            normals[i + j*3 + 2] = normal.z;  // z component of normal
        }
    }
    
    return normals;
}