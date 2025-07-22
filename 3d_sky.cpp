#include <3d_rendering.h>

void Renderer3D::init_sky() {
    // Create a skybox cube vertices
    float sky_vertices[] = {
        // Front face
        -1.0f, -1.0f,  1.0f,
         1.0f, -1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
        -1.0f,  1.0f,  1.0f,
        
        // Back face
        -1.0f, -1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,
         1.0f,  1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,
        
        // Top face
        -1.0f,  1.0f, -1.0f,
        -1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f,  1.0f, -1.0f,
        
        // Bottom face
        -1.0f, -1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,
         1.0f, -1.0f,  1.0f,
        -1.0f, -1.0f,  1.0f,
        
        // Right face
         1.0f, -1.0f, -1.0f,
         1.0f,  1.0f, -1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f, -1.0f,  1.0f,
        
        // Left face
        -1.0f, -1.0f, -1.0f,
        -1.0f, -1.0f,  1.0f,
        -1.0f,  1.0f,  1.0f,
        -1.0f,  1.0f, -1.0f
    };
    
    unsigned int sky_indices[] = {
        // Front face
        0,  1,  2,   2,  3,  0,
        // Back face
        4,  5,  6,   6,  7,  4,
        // Top face
        8,  9, 10,  10, 11,  8,
        // Bottom face
        12, 13, 14,  14, 15, 12,
        // Right face
        16, 17, 18,  18, 19, 16,
        // Left face
        20, 21, 22,  22, 23, 20
    };
    
    // Generate and bind VAO
    glGenVertexArrays(1, &sky_vao);
    glGenBuffers(1, &sky_vbo);
    glGenBuffers(1, &sky_ebo);
    
    glBindVertexArray(sky_vao);
    
    // Upload vertex data
    glBindBuffer(GL_ARRAY_BUFFER, sky_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(sky_vertices), sky_vertices, GL_STATIC_DRAW);
    
    // Upload index data
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sky_ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(sky_indices), sky_indices, GL_STATIC_DRAW);
    
    // Set vertex attributes
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    // Unbind
    glBindVertexArray(0);
    
    // Create shader program
    sky_shader = create_shader_program(sky_vertex_shader, sky_fragment_shader);
}

void Renderer3D::render_sky() {
    // Disable depth writing but keep depth testing for early z-reject
    glDepthMask(GL_FALSE);
    glDepthFunc(GL_LEQUAL);
    
    glUseProgram(sky_shader);
    
    // Create view and projection matrices
    glm::mat4 view = glm::lookAt(camera_pos, camera_pos + camera_front, camera_up);
    glm::mat4 projection = glm::perspective(glm::radians(45.0f), 1024.0f/768.0f, 0.1f, 10000.0f);
    
    // Send matrices to shader
    glUniformMatrix4fv(glGetUniformLocation(sky_shader, "view"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(sky_shader, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
    
    // Set sky colors (improved gradient)
    glUniform3f(glGetUniformLocation(sky_shader, "bottomColor"), 0.6f, 0.8f, 1.0f); // Light blue horizon
    glUniform3f(glGetUniformLocation(sky_shader, "topColor"), 0.1f, 0.3f, 0.8f);    // Deep blue zenith
    
    // Render skybox
    glBindVertexArray(sky_vao);
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
    
    // Restore depth settings
    glDepthMask(GL_TRUE);
    glDepthFunc(GL_LESS);
}