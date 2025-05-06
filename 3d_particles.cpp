#include "3d_rendering.h"

// particle system constants - these define the behavior of the rain
const float DEFAULT_PARTICLE_SIZE = 1.5f;
const float DEFAULT_RAIN_INTENSITY = 0.5f;
const int DEFAULT_MAX_PARTICLES = 3000;
const float PARTICLE_SPAWN_RADIUS = 300.0f;
const float PARTICLE_VISIBLE_RADIUS = 500.0f;
const float MIN_PARTICLE_HEIGHT = -10.0f;
const float MAX_PARTICLE_HEIGHT = 200.0f;
const float MIN_PARTICLE_LIFETIME = 1.0f;
const float MAX_PARTICLE_LIFETIME = 3.0f;
const float RAIN_VELOCITY_BASE = -10.0f;
const float RAIN_VELOCITY_VARIANCE = 2.0f;

// initialize particle system with shaders - setup opengl components for rendering rain
void Renderer3D::setup_particle_system() {
    GLuint particle_vertex = compile_shader(GL_VERTEX_SHADER, particle_vertex_shader);      // processes each particle's position/attributes
    GLuint particle_fragment = compile_shader(GL_FRAGMENT_SHADER, particle_fragment_shader);// determines each pixel's final color
    particle_shader_program = glCreateProgram();  // creates empty shader program container
    glAttachShader(particle_shader_program, particle_vertex); // adds vertex shader to program
    glAttachShader(particle_shader_program, particle_fragment); // adds fragment shader to program
    glLinkProgram(particle_shader_program); // combines shaders into executable program
    
    // verify shader linking - ensures shaders work together properly
    GLint success;
    glGetProgramiv(particle_shader_program, GL_LINK_STATUS, &success);
    if (!success) {
        char info_log[512];
        glGetProgramInfoLog(particle_shader_program, 512, nullptr, info_log);
        std::cerr << "particle shader program linking failed: " << info_log << std::endl;
        throw std::runtime_error("particle shader program linking failed");
    }
    glDeleteShader(particle_vertex);
    glDeleteShader(particle_fragment);
    init_rain_particles();
    std::cout << "Particle system initialized successfully" << std::endl;
}

// create initial rain particles
void Renderer3D::init_rain_particles() {
    rain_particles.clear(); // remove any current particles
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> pos_x_dist(-render_radius, render_radius); // randomize x positions
    std::uniform_real_distribution<float> pos_z_dist(-render_radius, render_radius); // randomize z positions
    std::uniform_real_distribution<float> height_dist(50.0f, MAX_PARTICLE_HEIGHT); // randomize spawn heights
    std::uniform_real_distribution<float> life_dist(MIN_PARTICLE_LIFETIME, MAX_PARTICLE_LIFETIME); // randomize lifetimes
    int particle_count = static_cast<int>(max_rain_particles * rain_intensity);
    rain_particles.reserve(particle_count);
    for (int i = 0; i < particle_count; i++) {
        rain_particle particle;
        particle.position = glm::vec3(pos_x_dist(gen), height_dist(gen), pos_z_dist(gen));
        // set velocity - mostly downward with slight horizontal randomness
        particle.velocity = glm::vec3(
            (gen() % 100) / 1000.0f - 0.05f, // slight sideways drift (-0.05 to 0.05)
            RAIN_VELOCITY_BASE - (gen() % 100) / 50.0f, // downward speed with variance
            (gen() % 100) / 1000.0f - 0.05f // slight forward/backward drift
        );
        particle.life = life_dist(gen); // total lifetime
        particle.age = (gen() % 100) / 100.0f * particle.life; // random initial age (creates natural distribution)
        rain_particles.push_back(particle);
    }
    std::cout << "Initialized " << rain_particles.size() << " rain particles" << std::endl;
}

// update particle positions and properties each frame
void Renderer3D::update_rain_particles(float delta_time) {
    if (rain_particles.empty() || delta_time > 0.1f) return;  // skip if no particles or if time jump is too large
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> pos_x_dist(-PARTICLE_SPAWN_RADIUS, PARTICLE_SPAWN_RADIUS);
    std::uniform_real_distribution<float> pos_z_dist(-PARTICLE_SPAWN_RADIUS, PARTICLE_SPAWN_RADIUS);
    std::uniform_real_distribution<float> height_dist(100.0f, MAX_PARTICLE_HEIGHT);
    std::uniform_real_distribution<float> life_dist(MIN_PARTICLE_LIFETIME, MAX_PARTICLE_LIFETIME);
    
    // update each particle
    for (auto& particle : rain_particles) {
        particle.age += delta_time; // increment age
        
        // check if particle needs respawning - when it's expired or below ground
        if (particle.age >= particle.life || particle.position.y < MIN_PARTICLE_HEIGHT) {
            // respawn near camera position
            particle.position = glm::vec3(
                camera_pos.x + (gen() % 100) / 50.0f * PARTICLE_SPAWN_RADIUS - PARTICLE_SPAWN_RADIUS / 2.0f,
                camera_pos.y + height_dist(gen),
                camera_pos.z + (gen() % 100) / 50.0f * PARTICLE_SPAWN_RADIUS - PARTICLE_SPAWN_RADIUS / 2.0f
            );
            // reset velocity with wind variation
            float wind_x = (gen() % 100) / 1000.0f - 0.05f;
            float wind_z = (gen() % 100) / 1000.0f - 0.05f;
            particle.velocity = glm::vec3(wind_x, RAIN_VELOCITY_BASE - (gen() % 100) / 50.0f, wind_z);
            
            // reset lifetime properties
            particle.life = life_dist(gen);
            particle.age = 0.0f;
        } else {
            // normal update - move particle based on velocity
            particle.position += particle.velocity * delta_time;
        }
    }
    
    // adjust particle count if intensity changed
    int desired_particle_count = static_cast<int>(max_rain_particles * rain_intensity);
    if (static_cast<int>(rain_particles.size()) != desired_particle_count) {
        if (rain_particles.size() < desired_particle_count) {
            // add more particles
            int particles_to_add = desired_particle_count - rain_particles.size();
            for (int i = 0; i < particles_to_add; i++) {
                rain_particle particle;
                
                // spawn around camera
                particle.position = glm::vec3(
                    camera_pos.x + (gen() % 100) / 50.0f * PARTICLE_SPAWN_RADIUS - PARTICLE_SPAWN_RADIUS / 2.0f,
                    camera_pos.y + height_dist(gen),
                    camera_pos.z + (gen() % 100) / 50.0f * PARTICLE_SPAWN_RADIUS - PARTICLE_SPAWN_RADIUS / 2.0f
                );
                
                // set velocity with wind variation
                float wind_x = (gen() % 100) / 1000.0f - 0.05f;
                float wind_z = (gen() % 100) / 1000.0f - 0.05f;
                particle.velocity = glm::vec3(wind_x, RAIN_VELOCITY_BASE - (gen() % 100) / 50.0f, wind_z);
                
                // set lifetime
                particle.life = life_dist(gen);
                particle.age = (gen() % 100) / 100.0f * particle.life;
                
                rain_particles.push_back(particle);
            }
        } else {
            // remove excess particles
            rain_particles.resize(desired_particle_count);
        }
        
        std::cout << "Adjusted rain particles: " << rain_particles.size() << std::endl;
    }
}

// render all rain particles to screen
void Renderer3D::render_rain_particles() {
    if (rain_particles.empty() || rain_intensity <= 0.01f) return;  // skip if no rain
    // setup opengl state for particle rendering
    glEnable(GL_DEPTH_TEST); // enable depth comparison for 3d positioning
    glDepthMask(GL_FALSE); // prevent particles from writing to depth buffer (allows overlap)
    glEnable(GL_BLEND); // enable transparency blending
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); // standard alpha blending
    glEnable(GL_PROGRAM_POINT_SIZE); // allow shader to control point size
    
    // activate particle shader
    glUseProgram(particle_shader_program);
    
    // setup camera transformations
    glm::mat4 view = glm::lookAt(camera_pos, camera_pos + camera_front, camera_up);
    glm::mat4 projection = glm::perspective(glm::radians(45.0f), 1024.0f/768.0f, 0.1f, 10000.0f);
    glUniformMatrix4fv(glGetUniformLocation(particle_shader_program, "view"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(particle_shader_program, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
    
    // set particle appearance
    glUniform1f(glGetUniformLocation(particle_shader_program, "particleSize"), particle_size);
    
    // gather particle data for rendering (cull distant particles for performance)
    std::vector<float> positions, velocities, ages, lives;
    positions.reserve(rain_particles.size() * 3);
    velocities.reserve(rain_particles.size() * 3);
    ages.reserve(rain_particles.size());
    lives.reserve(rain_particles.size());
    
    for (const auto& particle : rain_particles) {
        // distance culling - only render particles within view distance
        float dist_sq = (particle.position.x - camera_pos.x) * (particle.position.x - camera_pos.x) +
                       (particle.position.z - camera_pos.z) * (particle.position.z - camera_pos.z);
        if (dist_sq > PARTICLE_VISIBLE_RADIUS * PARTICLE_VISIBLE_RADIUS) continue;  // skip distant particles
        // collect particle data
        positions.insert(positions.end(), {particle.position.x, particle.position.y, particle.position.z});
        velocities.insert(velocities.end(), {particle.velocity.x, particle.velocity.y, particle.velocity.z});
        ages.push_back(particle.age);
        lives.push_back(particle.life);
    }
    
    if (positions.empty()) {
        // no visible particles - cleanup and return
        glDepthMask(GL_TRUE);
        glDisable(GL_BLEND);
        glDisable(GL_PROGRAM_POINT_SIZE);
        return;
    }
    
    // setup vertex attributes - temporary vao and vbos for this frame
    GLuint temp_vao, vbo_position, vbo_velocity, vbo_age, vbo_life;
    glGenVertexArrays(1, &temp_vao);
    glGenBuffers(1, &vbo_position);
    glGenBuffers(1, &vbo_velocity);
    glGenBuffers(1, &vbo_age);
    glGenBuffers(1, &vbo_life);
    glBindVertexArray(temp_vao);
    
    // position attribute (location 0)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_position);
    glBufferData(GL_ARRAY_BUFFER, positions.size() * sizeof(float), positions.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    // velocity attribute (location 1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_velocity);
    glBufferData(GL_ARRAY_BUFFER, velocities.size() * sizeof(float), velocities.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    
    // age attribute (location 2)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_age);
    glBufferData(GL_ARRAY_BUFFER, ages.size() * sizeof(float), ages.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)0);
    glEnableVertexAttribArray(2);
    
    // life attribute (location 3)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_life);
    glBufferData(GL_ARRAY_BUFFER, lives.size() * sizeof(float), lives.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)0);
    glEnableVertexAttribArray(3);
    
    // render particles as points
    glDrawArrays(GL_POINTS, 0, positions.size() / 3);
    
    // cleanup resources
    glBindVertexArray(0);
    glDeleteVertexArrays(1, &temp_vao);
    glDeleteBuffers(1, &vbo_position);
    glDeleteBuffers(1, &vbo_velocity);
    glDeleteBuffers(1, &vbo_age);
    glDeleteBuffers(1, &vbo_life);
    
    // restore opengl state
    glDepthMask(GL_TRUE);
    glDisable(GL_PROGRAM_POINT_SIZE);
    glDisable(GL_BLEND);
}

// // utility function to enhance rain visibility for testing
// void Renderer3D::enhance_rain_particles() {
//     particle_size = 5.0f;  // make particles larger for better visibility
    
//     // ensure minimum rain intensity
//     if (rain_intensity < 0.5f) {
//         rain_intensity = 0.5f;
//         std::cout << "Rain intensity increased to " << rain_intensity << std::endl;
//     }
    
//     // ensure sufficient particle count
//     if (max_rain_particles < 5000) {
//         max_rain_particles = 5000;
//         std::cout << "Max rain particles increased to " << max_rain_particles << std::endl;
//     }
    
//     // reset particles with new settings
//     init_rain_particles();
    
//     std::cout << "Rain particles enhanced for better visibility" << std::endl;
// }