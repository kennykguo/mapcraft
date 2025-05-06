#include <3d_rendering.h>


// need to update this with sunset color sky
void Renderer3D::init_sky() {
    // Create a fullscreen quad for sky
    float sky_vertices[] = {
        -1.0f, -1.0f, 0.0f,  // bottom left
         1.0f, -1.0f, 0.0f,  // bottom right
        -1.0f,  1.0f, 0.0f,  // top left
         1.0f,  1.0f, 0.0f   // top right
    };
    glGenVertexArrays(1, &sky_vao);
    glGenBuffers(1, &sky_vbo);
    glBindVertexArray(sky_vao);
    glBindBuffer(GL_ARRAY_BUFFER, sky_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(sky_vertices), sky_vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    sky_shader = create_shader_program(sky_vertex_shader, sky_fragment_shader);
}

void Renderer3D::render_sky() {
    glDisable(GL_DEPTH_TEST);
    glUseProgram(sky_shader);
    // Set sunset colors
    glUniform3f(glGetUniformLocation(sky_shader, "bottomColor"), 0.5f, 0.7f, 1.0f); // Orange-red
    glUniform3f(glGetUniformLocation(sky_shader, "topColor"), 0.1f, 0.3f, 0.8f);    // Deep blue
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    glUniform1f(glGetUniformLocation(sky_shader, "screenHeight"), (float)height);
    glBindVertexArray(sky_vao);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glEnable(GL_DEPTH_TEST);
}