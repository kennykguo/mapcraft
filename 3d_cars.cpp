#include <3d_rendering.h>

// hyperparameters as constants
constexpr float CAR_DENSITY = 0.5f; // controls the number of cars generated (lower = fewer cars)
constexpr float CAR_SPACING = 20.0f; // fixed value for spacing between cars
constexpr float MIN_SEGMENT_LENGTH = 10.0f; // minimum road segment length to consider
constexpr int MAX_CARS_PER_SEGMENT = 20; // maximum cars per segment regardless of length
constexpr float SKIP_PROBABILITY = 0.4f; // probability threshold for skipping car placement
constexpr float SEGMENT_MIN_POS = 0.15f; // minimum position along segment (avoid edges)
constexpr float SEGMENT_MAX_POS = 0.85f; // maximum position along segment (avoid edges)
constexpr float LANE_WIDTH_FACTOR = 0.35f; // factor to determine lane offset from road width
constexpr float CAR_HEIGHT = 2.0f; // height of car above the road
constexpr int CAR_TYPE_COUNT = 4; // number of different car types

extern const float FOV_DEGREES;
extern const float ASPECT_RATIO;
extern const float NEAR_PLANE;
extern const float FAR_PLANE;

constexpr float AMBIENT = 0.3f; // ambient light intensity
constexpr float DIFFUSE = 0.7f; // diffuse light intensity
constexpr float SPECULAR = 0.5f; // specular light intensity
constexpr float SHININESS = 32.0f; // material shininess
constexpr float CAR_SCALE_X = 3.0f; // car x scale
constexpr float CAR_SCALE_Y = 4.0f; // car y scale
constexpr float CAR_SCALE_Z = 4.0f; // car z scale

// randomly populates road segments with cars based on segment length and predefined density parameters
// logic spacing logic to avoid overpopulation, calculates appropriate positions along road segments with lane offsets, and aligns cars with road direction
// skips segments that are too short and uses probability dist to create natural-looking traffic patterns
void Renderer3D::generate_cars() {
    std::cout << "generating car meshes..." << std::endl;
    cars.clear(); // clear previously generated cars

    // initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // iterate through road segments
    for (const auto& road : road_segments) {
        // init random distributions for car generation
        std::uniform_real_distribution<float> skip_dist(0.0f, 1.0f); // determines if a car is placed
        std::uniform_int_distribution<int> car_type_dist(0, CAR_TYPE_COUNT - 1); // selects car type
        
        // for each road segment
        for (int i = 0; i < road.vertex_count - 1; i++) {
            glm::vec3 p1 = road.vertices[i];
            glm::vec3 p2 = road.vertices[i + 1];
            glm::vec3 dir = glm::normalize(p2 - p1); // compute direction vector along the road
            
            // skip very short road segments
            float segment_length = glm::distance(glm::vec2(p1.x, p1.z), glm::vec2(p2.x, p2.z));
            if (segment_length < MIN_SEGMENT_LENGTH) continue; 

            // calculate number of cars for this segment
            int max_cars = std::min(static_cast<int>(segment_length / CAR_SPACING), MAX_CARS_PER_SEGMENT);
            int num_cars = static_cast<int>(max_cars * CAR_DENSITY); // tuned furhter by a hyperparameter

            // place cars sampled from dist on this street segment
            for (int j = 0; j < num_cars; j++) {
                // randomly skip some cars to avoid overpopulation
                if (skip_dist(gen) > SKIP_PROBABILITY) continue; 

                // compute random position along the segment
                std::uniform_real_distribution<float> pos_dist(SEGMENT_MIN_POS, SEGMENT_MAX_POS);
                float t = pos_dist(gen);
                // place it somewhere along closer to the middle dependening on the sampled value of t
                glm::vec3 pos = p1 + dir * (t * segment_length);

                // offset to position the car within the correct lane
                float lane_offset = road.width * LANE_WIDTH_FACTOR;
                int lane = std::uniform_int_distribution<int>(0, 1)(gen);
                glm::vec3 perp = glm::normalize(glm::cross(dir, glm::vec3(0, 1, 0))); // right hand rule?
                pos += (lane == 0) ? perp * lane_offset : -perp * lane_offset;
                pos.y = CAR_HEIGHT; // elevate car slightly above the road

                // create and store car mesh data
                car_mesh car;
                car.position = pos;
                car.rotation = glm::degrees(atan2(dir.z, dir.x)); // align car with road direction
                car.type = car_type_dist(gen);
                cars.push_back(car);
            }
        }
    }
    std::cout << "generated " << cars.size() << " cars" << std::endl;
}



// activates phong shader program for 3d rendering
// creates view and projection matrices for camera perspective
// sets up lighting parameters and camera position
// configures vertex buffer objects for mesh data
// sets material properties for surface appearance
// disables textures for car meshes
// for each car:
// creates model matrix with position, rotation and scale
// selects color based on car type (red, blue, black or gray)
// renders the car mesh using triangles
// cleans up gpu resources after rendering
/*
 * 1. shader activation
 * 2. camera and projection setup
 * 3. buffer creation and data upload
 * 4. material property configuration
 * 5. per-object rendering with transformations
 * 6. OpenGL resource cleanup
 */
void Renderer3D::draw_cars() {
    // check if there are any cars to draw
    if (cars.empty()) return;  // exit early if there are no cars to draw

    std::vector<float> car_vertices;
    std::vector<float> car_normals;
    
    // SHADER ACTIVATION
    // activate the phong shading shader program
    // tells gpu to use these specific instructions from phong shaders for processing geometry so they can execute on the data
    // a shader contains two main components - vertex shader and a fragment shader
    // vertex shader - tells the gpu how to position each vertex in 3d space
    // fragment shader - tells the gpu how to color each pixel
    
    // glUseProgram - activates a shader program for rendering
    // - takes a GLuint program ID and makes it the currently active shader
    // - all subsequent draw calls will use this shader for rendering
    // - this is where we tell the GPU which instructions to use for processing vertices and fragments
    glUseProgram(phong_shader_program);
    
    // CAMERA TRANSFORMATIONS
    // create view and projection matrices for 3D rendering
    // - view matrix - transforms world coordinates to camera's view
    // - projection matrix - projects 3D coordinates to 2D screen space
    glm::mat4 view = glm::lookAt(camera_pos, camera_pos + camera_front, camera_up);
    glm::mat4 projection = glm::perspective(
        glm::radians(FOV_DEGREES), 
        ASPECT_RATIO, 
        NEAR_PLANE, 
        FAR_PLANE
    );
    
    // SEND MATRICES TO SHADER
    // glGetUniformLocation - finds the location of a uniform variable in the shader
    // - takes program ID and uniform name, returns location ID (GLint)
    // - returns -1 if the uniform doesn't exist or isn't used
    // glUniformMatrix4fv: Uploads a 4x4 matrix to a shader uniform
    // - parameters - location, count, transpose flag, pointer to matrix data
    // - allows changing shader behavior without recompiling
    // pass these matrices to shader - gives gpu information about how to transform and give the scene lighting
    // glGetUniformLocation - asks gpu where should i put the data so the shader can find it?
    // glUniformMatrix4fv - sends data to that location
    // takes in location to store data, count of matrices, and where to transpose matrix, and convert matrix to format gpu understands
    glUniformMatrix4fv(glGetUniformLocation(phong_shader_program, "view"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(phong_shader_program, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
    
    // SEND LIGHTING PARAMETERS
    // glUniform3fv -ploads a 3-component vector to a shader uniform
    // - similar to matrix version but for vec3 type
    // - these values control the lighting calculations in the fragment shader
    glUniform3fv(glGetUniformLocation(phong_shader_program, "viewPos"), 1, glm::value_ptr(camera_pos));
    glUniform3fv(glGetUniformLocation(phong_shader_program, "lightPos"), 1, glm::value_ptr(light_position));
    glUniform3fv(glGetUniformLocation(phong_shader_program, "lightColor"), 1, glm::value_ptr(light_color));
    
    // CREATE GPU BUFFERS
    // create and configure vertex buffers
    // tells gpu where to find the mesh data
    // create storage spach on the GPU for our vertex data
    // vertex array object stores the organization
    // vertex buffer object stores the actual vertex data
    // these are stored on the gpu and can be referenced later
    // binding a vao - all following vertex configs should apply
    // glGenVertexArrays - creates Vertex Array Objects (VAOs)
    // - VAOs store the configuration of vertex attribute data
    // - they remember which VBOs are bound and how data is formatted
    // glGenBuffers - creates buffer objects (VBOs)
    // - VBOs store vertex data on the GPU
    // - multiple types of data can be stored in different VBOs (positions, normals, etc.)
    GLuint car_vao, vbo_position, vbo_normal;
    glGenVertexArrays(1, &car_vao);
    glGenBuffers(1, &vbo_position);
    glGenBuffers(1, &vbo_normal);
    
    // glBindVertexArray - selects a VAO to configure
    // all vertex attribute configurations stored in this VAO
    glBindVertexArray(car_vao);
    
    // UPLOAD VERTEX POSITION DATA
    // glBindBuffer - Selects a buffer for operations
    // - GL_ARRAY_BUFFER is for vertex attributes
    // - other types include GL_ELEMENT_ARRAY_BUFFER for indices
    glBindBuffer(GL_ARRAY_BUFFER, vbo_position);
    
    // glBufferData - allocates and fills buffer with data
    // - creates and initializes a buffer object's data store
    // - parameters - target, size in bytes, data pointer, usage hint
    // - GL_STATIC_DRAW hint tells OpenGL the data won't change often
    glBufferData(GL_ARRAY_BUFFER, car_vertices.size() * sizeof(float), car_vertices.data(), GL_STATIC_DRAW);
    
    // glVertexAttribPointer - defines how to interpret buffer data
    // - parameters - attribute index, size (components per vertex), type, normalized flag, 
    //   stride (bytes between vertices), offset (bytes to first component)
    // - This tells OpenGL how to feed the buffer data to the shader inputs
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    
    // glEnableVertexAttribArray - enables a vertex attribute
    // - by default, vertex attributes are disabled
    // - must be enabled for the shader to access the data
    glEnableVertexAttribArray(0);
    
    // UPLOAD VERTEX NORMAL DATA
    // same process for normal data - bind, upload, configure, enable
    glBindBuffer(GL_ARRAY_BUFFER, vbo_normal);
    glBufferData(GL_ARRAY_BUFFER, car_normals.size() * sizeof(float), car_normals.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    
    // SET MATERIAL PROPERTIES
    // glUniform1f - Sets a float uniform in the shader
    // - these control how light interacts with the surface
    // - material properties affect the Phong lighting model calculation
    glUniform1f(glGetUniformLocation(phong_shader_program, "ambient"), AMBIENT);
    glUniform1f(glGetUniformLocation(phong_shader_program, "diffuse"), DIFFUSE);
    glUniform1f(glGetUniformLocation(phong_shader_program, "specular"), SPECULAR);
    glUniform1f(glGetUniformLocation(phong_shader_program, "shininess"), SHININESS);
    
    // CONFIGURE TEXTURING
    // glUniform1i - sets an integer uniform in the shader
    // used here to enable/disable texture features
    glUniform1i(glGetUniformLocation(phong_shader_program, "useTexture"), 0);
    glUniform1i(glGetUniformLocation(phong_shader_program, "useWindows"), 0);
    
    // loop through each car and render it with a specific transform and color
    for (const auto& car : cars) {
        // create model matrix for positioning, rotating, and scaling
        // - transformations are applied in reverse order - scale, then rotate, then translate
        glm::mat4 model = glm::mat4(1.0f); // start with identity matrix
        model = glm::translate(model, car.position); // move to position
        model = glm::rotate(model, glm::radians(car.rotation), glm::vec3(0.0f, 1.0f, 0.0f)); // rotate around Y axis
        model = glm::scale(model, glm::vec3(CAR_SCALE_X, CAR_SCALE_Y, CAR_SCALE_Z)); // scale to size
        // send model matrix to shader
        glUniformMatrix4fv(glGetUniformLocation(phong_shader_program, "model"), 1, GL_FALSE, glm::value_ptr(model));
        
        // select a color based on car type
        glm::vec3 car_color;
        switch (car.type % CAR_TYPE_COUNT) {
            case 0: car_color = glm::vec3(0.8f, 0.1f, 0.1f); break; // red
            case 1: car_color = glm::vec3(0.1f, 0.1f, 0.8f); break; // blue
            case 2: car_color = glm::vec3(0.1f, 0.1f, 0.1f); break; // black
            default: car_color = glm::vec3(0.8f, 0.8f, 0.8f); break; // gray
        }
        // send color to shader
        glUniform3fv(glGetUniformLocation(phong_shader_program, "objectColor"), 1, glm::value_ptr(car_color));
        
        // glDrawArrays - renders primitives from array data
        // - parameters - primitive type, starting index, vertex count
        // - GL_TRIANGLES means every 3 vertices form a triangle
        glDrawArrays(GL_TRIANGLES, 0, car_vertices.size() / 3);
        // - this executes the entire OpenGL pipeline - vertex shader → fragment shader → screen
    }


    // glDeleteVertexArrays/glDeleteBuffers: Release GPU resources
    // - Prevents memory leaks in the GPU
    // - Good practice to clean up objects when done with them
    glDeleteVertexArrays(1, &car_vao);
    glDeleteBuffers(1, &vbo_position);
    glDeleteBuffers(1, &vbo_normal);
}


// Creates a cylinder mesh in the given vertex and normal arrays
// This function demonstrates geometry generation for OpenGL rendering:
// Creates triangular faces for the cylinder sides
// Generates normals for proper lighting
// Organizes vertices in triangle order for GL_TRIANGLES rendering
    // vertices - vector to store vertex positions (x,y,z triplets)
    // normals - vector to store normal vectors (x,y,z triplets)
    // center - center point of the cylinder
    // radius - radius of the cylinder
    // width - width/length of the cylinder
    // segments - number of segments around the circumference (more = smoother)
void Renderer3D::add_cylinder_mesh(std::vector<float>& vertices, std::vector<float>& normals,  const glm::vec3& center, float radius, float width, int segments) { 
    // for a wheel, the cylinder axis is along the z-axis
    // to create a cylinder, we approximate it with circulating triangular prisms
    for (int i = 0; i < segments; i++) {

        // segment angle
        float angle1 = glm::radians(static_cast<float>(i) / segments * 360.0f);
        float angle2 = glm::radians(static_cast<float>(i + 1) / segments * 360.0f);
        
        // approximate the coordinates
        float x1 = radius * cos(angle1);
        float y1 = radius * sin(angle1);
        float x2 = radius * cos(angle2);
        float y2 = radius * sin(angle2);
        
        // calculate normals for the cylinder sides (pointing in the direction from center to each point)
        glm::vec3 normal1(x1, y1, 0.0f);
        normal1 = glm::normalize(normal1);
        glm::vec3 normal2(x2, y2, 0.0f);
        normal2 = glm::normalize(normal2);
        
        // cylinder  rim edge - first triangle - cylindrical surface side as a rectangle. we're just drawing a single triangle for now
        vertices.push_back(center.x + x1);
        vertices.push_back(center.y + y1);
        vertices.push_back(center.z - width/2);
        vertices.push_back(center.x + x2);
        vertices.push_back(center.y + y2);
        vertices.push_back(center.z - width/2);
        vertices.push_back(center.x + x2);
        vertices.push_back(center.y + y2);
        vertices.push_back(center.z + width/2);
        // add normals for first triangle // circular end cap
        normals.push_back(normal1.x);
        normals.push_back(normal1.y);
        normals.push_back(normal1.z);
        normals.push_back(normal2.x);
        normals.push_back(normal2.y);
        normals.push_back(normal2.z);
        normals.push_back(normal2.x);
        normals.push_back(normal2.y);
        normals.push_back(normal2.z);
        
        // cylinder  rim edge - second triangle - cylindrical surface side as a rectangle. we're just drawing another single triangle for now
        vertices.push_back(center.x + x1);
        vertices.push_back(center.y + y1);
        vertices.push_back(center.z - width/2);
        vertices.push_back(center.x + x2);
        vertices.push_back(center.y + y2);
        vertices.push_back(center.z + width/2);
        vertices.push_back(center.x + x1);
        vertices.push_back(center.y + y1);
        vertices.push_back(center.z + width/2);
        // add normals for second triangle
        normals.push_back(normal1.x);
        normals.push_back(normal1.y);
        normals.push_back(normal1.z);
        normals.push_back(normal2.x);
        normals.push_back(normal2.y);
        normals.push_back(normal2.z);
        normals.push_back(normal1.x);
        normals.push_back(normal1.y);
        normals.push_back(normal1.z);
        
        // wheel face - front - normal points down
        vertices.push_back(center.x);
        vertices.push_back(center.y);
        vertices.push_back(center.z - width/2);
        vertices.push_back(center.x + x1);
        vertices.push_back(center.y + y1);
        vertices.push_back(center.z - width/2);
        vertices.push_back(center.x + x2);
        vertices.push_back(center.y + y2);
        vertices.push_back(center.z - width/2);
        // front face normals (pointing in negative z)
        for (int j = 0; j < 3; j++) {
            normals.push_back(0.0f);
            normals.push_back(0.0f);
            normals.push_back(-1.0f);
        }
        
        // wheel face - back - normal points up
        vertices.push_back(center.x);
        vertices.push_back(center.y);
        vertices.push_back(center.z + width/2);
        vertices.push_back(center.x + x2);
        vertices.push_back(center.y + y2);
        vertices.push_back(center.z + width/2);
        vertices.push_back(center.x + x1);
        vertices.push_back(center.y + y1);
        vertices.push_back(center.z + width/2);
        // back face normals (pointing in positive z)
        for (int j = 0; j < 3; j++) {
            normals.push_back(0.0f);
            normals.push_back(0.0f);
            normals.push_back(1.0f);
        }
    }
}