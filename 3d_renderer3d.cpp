#include "3d_rendering.h"

// rendering constants - these define core rendering parameters
const float INITIAL_CAMERA_HEIGHT = 150.0f;
const float INITIAL_CAMERA_DISTANCE = 300.0f;
const glm::vec3 INITIAL_CAMERA_POS = glm::vec3(0.0f, INITIAL_CAMERA_HEIGHT, INITIAL_CAMERA_DISTANCE);
const glm::vec3 INITIAL_CAMERA_FRONT = glm::vec3(0.0f, 0.0f, -1.0f);
const glm::vec3 INITIAL_CAMERA_UP = glm::vec3(0.0f, 1.0f, 0.0f);
const float INITIAL_YAW = -90.0f;
const float INITIAL_PITCH = 0.0f;
const float INITIAL_FRAME_TIME = 0.0f;
const float DEFAULT_RENDER_RADIUS = 2500.0f;
const int DEFAULT_MAX_RAIN_PARTICLES = 10000;
const float DEFAULT_TESSELLATION_LEVEL = 4.0f;
const int WINDOW_WIDTH = 1024;
const int WINDOW_HEIGHT = 768;
const float CAMERA_MOVEMENT_SPEED = 50.0f;
const float MOUSE_SENSITIVITY = 0.005f;
const float PITCH_CONSTRAINT_MAX = 89.0f;
const float PITCH_CONSTRAINT_MIN = -89.0f;

// initialize 3d renderer with geographic reference point - sets up entire rendering system
Renderer3D::Renderer3D(const latlon& initial_pos) {
    // set geographic reference - all world coordinates relative to this point
    reference_point = initial_pos;
    camera_pos = INITIAL_CAMERA_POS; // camera position in 3d space
    camera_front = INITIAL_CAMERA_FRONT; // direction camera is looking
    camera_up = INITIAL_CAMERA_UP; // up vector for camera orientation
    yaw = INITIAL_YAW; // horizontal rotation angle
    pitch = INITIAL_PITCH; // vertical rotation angle
    first_mouse = true; // flag for mouse input initialization
    last_frame = INITIAL_FRAME_TIME; // timing for frame-based calculations
    last_x = 0.0; // previous mouse x position
    last_y = 0.0; // previous mouse y position
    render_radius = DEFAULT_RENDER_RADIUS; // distance from camera to render
    d_building_data = nullptr;
    d_building_vertices = nullptr;
    building_count = 0;
    d_road_data = nullptr;
    d_road_vertices = nullptr;
    road_count = 0;
    d_frustum_planes = nullptr;
    d_visible_building_indices = nullptr;
    d_visible_building_count = nullptr;
    visible_building_count = 0;
    d_visible_road_indices = nullptr;
    d_visible_road_count = nullptr;
    visible_road_count = 0;
    d_natural_feature_data = nullptr;
    d_natural_feature_vertices = nullptr;
    d_visible_natural_feature_indices = nullptr;
    d_visible_natural_feature_count = nullptr;
    visible_natural_feature_count = 0;
    natural_feature_count = 0;
    terrain_vao = 0;
    terrain_vbo_position = 0;
    terrain_vbo_tex_coords = 0;
    terrain_ebo = 0;
    shadow_map_fbo = 0;
    shadow_map_texture = 0;
    phong_shader_program = 0;
    tess_shader_program = 0;
    water_shader_program = 0;
    shadow_shader_program = 0;
    terrain_shader_program = 0;
    particle_shader_program = 0;
    vao = 0;
    vbo = 0;
    shader_program = 0;
    max_rain_particles = DEFAULT_MAX_RAIN_PARTICLES; // maximum number of raindrops
    tessellation_level = DEFAULT_TESSELLATION_LEVEL; // terrain detail level
    std::cout << "initializing 3D Renderer..." << std::endl;
    // debug output - verifies reference point conversion
    std::cout << "Renderer3D - Using reference point: lat=" << reference_point.latitude  << ", lon=" << reference_point.longitude << std::endl;
    // test coordinate conversion - should place reference point at origin
    glm::vec3 origin_test = latlon_to_3d(reference_point, reference_point);
    std::cout << "Reference point 3D (should be ~0,0,0): " << origin_test.x << ", " << origin_test.y << ", " << origin_test.z << std::endl;
    
    // initialize glfw - sets up window and input system
    if (!glfwInit()) {
        throw std::runtime_error("Failed to initialize GLFW");
    }
    
    // configure opengl context - specifies version and profile
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3); // opengl 3.x
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3); // opengl x.3
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // core profile (modern opengl)
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // compatibility flag
    
    // create window - establishes rendering surface
    window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "3D Street View", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window");
    }
    
    // set window callbacks - handles window events
    glfwSetWindowCloseCallback(window, [](GLFWwindow* w) {  // lambda function for close event
        std::cout << "Window close requested" << std::endl;
    });
    
    // create opengl context - makes window active for rendering
    glfwMakeContextCurrent(window);
    
    // initialize glad - loads opengl function pointers
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        glfwTerminate();
        throw std::runtime_error("Failed to initialize GLAD");
    }
    
    // enable core opengl features - required for 3d rendering
    glEnable(GL_DEPTH_TEST);  // enables depth buffer for proper 3d object ordering
    
    // initialize rendering systems - prepares shaders and gpu resources
    setup_shaders();
    
    // create building rendering objects - persistent storage for building data
    glGenVertexArrays(1, &building_vao); // vertex array for attribute configuration
    glGenBuffers(1, &building_vbo_position); // buffer for position data
    glGenBuffers(1, &building_vbo_normal); // buffer for lighting normals
    glGenBuffers(1, &building_vbo_texcoord); // buffer for texture coordinates
    
    std::cout << "loading features and roads in Renderer construction:" << std::endl;
    
    // load geographic data - converts raw data to renderable geometry
    load_features(); // loads buildings and structures
    load_roads(); // loads street network
    load_natural_features(); // loads parks, water bodies, etc.
    
    // setup spatial acceleration - enables efficient culling
    init_cuda_resources(); // prepares gpu acceleration for visibility testing
    
    std::cout << "3D Renderer initialized successfully" << std::endl;
}

// free all resources
Renderer3D::~Renderer3D() {
    std::cout << "Shutting down 3D Renderer..." << std::endl;
    // cleanup core opengl objects - release basic rendering resources
    if (vao) glDeleteVertexArrays(1, &vao); // delete vertex array if allocated
    if (vbo) glDeleteBuffers(1, &vbo); // delete vertex buffer if allocated
    if (shader_program) glDeleteProgram(shader_program); // delete shader program if created
    // cleanup cuda resources - release gpu acceleration structures
    cleanup_cuda_resources();
    // cleanup terrain resources - release ground rendering data
    if (terrain_vao) {
        glDeleteVertexArrays(1, &terrain_vao);
        glDeleteBuffers(1, &terrain_vbo_position);
        glDeleteBuffers(1, &terrain_vbo_tex_coords);
        glDeleteBuffers(1, &terrain_ebo);
    }
    // cleanup shadow mapping resources - release depth rendering buffers
    if (shadow_map_fbo) glDeleteFramebuffers(1, &shadow_map_fbo);
    if (shadow_map_texture) glDeleteTextures(1, &shadow_map_texture);
    // cleanup shader programs - release all gpu programs
    if (phong_shader_program) glDeleteProgram(phong_shader_program);
    if (tess_shader_program) glDeleteProgram(tess_shader_program);
    if (water_shader_program) glDeleteProgram(water_shader_program);
    if (shadow_shader_program) glDeleteProgram(shadow_shader_program);
    if (terrain_shader_program) glDeleteProgram(terrain_shader_program);
    if (particle_shader_program) glDeleteProgram(particle_shader_program);
    // cleanup window system - shutdown glfw environment
    glfwDestroyWindow(window);  // destroy rendering window
    glfwTerminate();            // shutdown glfw system
    
    std::cout << "3D Renderer destroyed" << std::endl;
}

// handle keyboard input
void Renderer3D::process_input(float delta_time) {
    // check for exit request - escape key closes application
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }
    // calculate frame-based speed - ensures consistent movement regardless of framerate
    float camera_speed = CAMERA_MOVEMENT_SPEED * delta_time;
    // wasd movement controls - standard fps-style navigation
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {  // forward movement
        camera_pos += camera_speed * camera_front;
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {  // backward movement
        camera_pos -= camera_speed * camera_front;
    }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {  // left strafe
        camera_pos -= glm::normalize(glm::cross(camera_front, camera_up)) * camera_speed;
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {  // right strafe
        camera_pos += glm::normalize(glm::cross(camera_front, camera_up)) * camera_speed;
    }
    // vertical movement controls - useful for architectural visualization
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {      // move up
        camera_pos += camera_up * camera_speed;
    }
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) { // move down
        camera_pos -= camera_up * camera_speed;
    }
}

// handle mouse input
void Renderer3D::mouse_callback(double xpos, double ypos) {
    // initialize mouse tracking - prevents camera jump on first mouse movement
    if (first_mouse) {
        last_x = xpos;
        last_y = ypos;
        first_mouse = false;
    }
    // calculate mouse movement delta - determines rotation amount
    float xoffset = xpos - last_x;
    float yoffset = last_y - ypos; // reversed because opengl y coordinates go bottom to top
    last_x = xpos;
    last_y = ypos;
    // apply sensitivity - scales raw mouse movement to camera rotation
    float sensitivity = MOUSE_SENSITIVITY;
    xoffset *= sensitivity;
    yoffset *= sensitivity;
    // update camera angles - yaw is horizontal, pitch is vertical
    yaw += xoffset; // horizontal rotation around vertical axis
    pitch += yoffset; // vertical rotation around horizontal axis
    // constrain pitch angle - prevents camera from flipping over
    if (pitch > PITCH_CONSTRAINT_MAX) {
        pitch = PITCH_CONSTRAINT_MAX;
    }
    if (pitch < PITCH_CONSTRAINT_MIN) {
        pitch = PITCH_CONSTRAINT_MIN;
    }
    // recalculate camera direction vector - converts euler angles to direction
    glm::vec3 front;
    front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));  // x component depends on both angles
    front.y = sin(glm::radians(pitch));                           // y component only depends on pitch
    front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));  // z component depends on both angles
    camera_front = glm::normalize(front);                         // normalize to maintain unit length
}