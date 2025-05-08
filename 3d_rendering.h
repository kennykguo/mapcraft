#ifndef _3D_RENDERING_H_
#define _3D_RENDERING_H_

/**
 * OpenGL Core API Documentation
 * -----------------------------
 * 
 * Key OpenGL Types:
 * - GLuint:  Unsigned integer, used for object handles/IDs (buffers, textures, etc.)
 * - GLint:   Signed integer, used for shader parameters and coordinates
 * - GLenum:  Enumeration type, used for constants that control OpenGL behavior
 * - GLfloat: Float type, used for coordinates, colors, etc.
 * - GLboolean: Boolean type, used for flags
 * 
 * Core OpenGL Objects:
 * - Vertex Array Objects (VAO): Store vertex attribute configurations
 * - Vertex Buffer Objects (VBO): Store vertex data (positions, normals, etc.)
 * - Element Buffer Objects (EBO): Store indices for indexed rendering
 * - Textures: Store image data for rendering
 * - Framebuffers: Custom render targets
 * - Shaders: Programs that run on the GPU
 * 
 * Render Pipeline:
 * 1. CPU prepares data and sends to GPU
 * 2. Vertex Shader: Transforms vertices
 * 3. Tessellation Control Shader (optional): Controls tessellation
 * 4. Tessellation Evaluation Shader (optional): Generates new vertices
 * 5. Geometry Shader (optional): Creates/modifies primitives
 * 6. Fragment Shader: Colors pixels
 * 7. Output merging: Depth testing, blending, etc.
 * 
 * Common Function Patterns:
 * - gl[Create/Generate][Object] - Creates OpenGL objects
 * - glBind[Object] - Makes an object active
 * - glDelete[Object] - Destroys OpenGL objects
 * - glUniform[Type] - Sets shader variables
 * - glDraw[Arrays/Elements] - Executes the rendering pipeline
 * 
 * Shader Functions:
 * - glCreateShader(): Creates a shader object
 * - glShaderSource(): Sets shader source code
 * - glCompileShader(): Compiles shader source
 * - glCreateProgram(): Creates a shader program
 * - glAttachShader(): Attaches shader to program
 * - glLinkProgram(): Links shader program
 * - glUseProgram(): Activates shader program
 * 
 * Buffer Functions:
 * - glGenBuffers(): Creates buffer objects
 * - glBindBuffer(): Selects a buffer for operations
 * - glBufferData(): Uploads data to buffer
 * - glVertexAttribPointer(): Defines vertex attribute layout
 * - glEnableVertexAttribArray(): Enables vertex attribute
 * 
 * Texture Functions:
 * - glGenTextures(): Creates texture objects
 * - glBindTexture(): Selects texture for operations
 * - glTexImage2D(): Uploads texture data
 * - glTexParameteri(): Sets texture parameters
 * 
 * Drawing Functions:
 * - glDrawArrays(): Draws primitives from array data
 * - glDrawElements(): Draws primitives from indexed data
 * 
 * Uniform Functions:
 * - glGetUniformLocation(): Gets uniform variable location
 * - glUniform*(): Sets uniform variable values
 * 
 * State Functions:
 * - glEnable()/glDisable(): Enables/disables capabilities
 * - glBlendFunc(): Sets blending function
 * - glDepthFunc(): Sets depth test function
 */
 

// GLAD for loading OpenGL functions
// GLFW for window creation and input handling
// GLM for 3D math operations
#include <glad/glad.h> 
#include <GLFW/glfw3.h>
#include <glm/glm.hpp> 
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <iostream>
#include <mutex>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <mutex>
#include <algorithm>
#include <random>

// project headers
#include "3d_rendering.cuh"
#include "cuda_spatial_grid.cuh"
#include "cuda_frustum_culling.cuh"
#include "3d_textures.h"
#include "3d_shaders.h"

// forward declarations for cuda interoperability
struct float3;
struct float4;
struct cuda_grid;
struct gpu_building_data;
struct gpu_road_data;
struct gpu_natural_feature_data;

struct point2d {
  double x, y;
};

struct rectangle {
  point2d min, max;
};

struct color {
  uint8_t red, green, blue, alpha;
};

// 3D vector structure with glm compatibility
struct vec3 {
  union {
      struct { float x, y, z; };
      glm::vec3 glm_vec; // this allows easy conversion with GLM vectors
  };

  // constructor from GLM vec3
  vec3(const glm::vec3& v) : x(v.x), y(v.y), z(v.z) {}
  
  // default constructor
  vec3(float _x = 0, float _y = 0, float _z = 0) : x(_x), y(_y), z(_z) {}
  
  // conversion to GLM vec3
  operator glm::vec3() const { return glm::vec3(x, y, z); }
};

struct latlon {
  float latitude;
  float longitude;

  // constructor
  latlon(float _latitude, float _longitude) {
      latitude = _latitude;
      longitude = _longitude;
  }

  latlon() {
    latitude = 0.0;
    longitude = 0.0;
  }

};

// street segment data
struct street_segment_data {
  uint32_t index;
  uint64_t osm_id;
  bool one_way;
  float speed_limit;
  uint32_t street_id;
  std::string street_name;
  std::string highway_type;
  int highway_type_index;
  std::vector<point2d> points;
  rectangle bounding_box;
  point2d centroid;
  float width;
  float height;
  float bounding_radius;
  std::vector<vec3> points3d;
  vec3 centroid3d;
};

// feature data
struct feature_data {
  uint32_t index;
  std::string type;
  std::string name;
  uint64_t osm_id;
  std::vector<point2d> points;
  rectangle bounding_box;
  bool is_closed;
  point2d centroid;
  double area;
  float height;
  float bounding_radius;
  std::vector<vec3> points3d;
  vec3 centroid3d;
  bool is_skyscraper;
  int roof_type;
  bool has_roof;
  color rgba;
};




//==============================================================================
// GLOBAL DATA STRUCTURES
//==============================================================================
extern std::vector<street_segment_data> graphics_street_segment_data;
extern std::vector<feature_data> graphics_feature_data;
extern latlon reference_point;

//==============================================================================
// EXTERNAL HELPER FUNCTIONS
//==============================================================================
bool load_street_segments(const std::string& filename, std::vector<street_segment_data>& out_data);
bool load_features(const std::string& filename, std::vector<feature_data>& out_data);
glm::vec3 color_to_vec3(const color& c);
glm::vec3 latlon_to_3d(const latlon& coord, const latlon& reference);
void start_3d_view(const latlon& position);
void start_3d_rendering(double latitude, double longitude);


//==============================================================================
// GEOMETRY DATA STRUCTURES
//==============================================================================
// basic building structure shared between host and device
struct building {
  glm::vec3* vertices; // array of vertices forming the building footprint
  int vertex_count;           // number of vertices
  float height;               // height of the building
  glm::vec3 centroid;         // center point of the building
  float bounding_radius;      // radius of bounding sphere for culling
  bool has_roof;              // whether the building has a distinct roof
  int roof_type;              // 0=flat, 1=gabled, 2=hipped, etc.
  glm::vec3 color;            // color for rendering
};

// road segment structure for 3D rendering
struct road_segment {
  glm::vec3* vertices;        // array of vertices forming the road centerline
  int vertex_count;           // number of vertices
  float width;                // width of the road
  float elevation;            // height above ground
  glm::vec3 centroid;         // center point of the road segment
  float bounding_radius;      // radius of bounding sphere for culling
  int road_type;              // type of road (highway, residential, etc.)
};

// natural features
struct natural_feature {
  glm::vec3* vertices;        // array of vertices forming the feature outline
  int vertex_count;           // number of vertices
  float elevation;            // height above/below ground
  glm::vec3 centroid;         // center point of the feature
  float bounding_radius;      // radius of bounding sphere for culling
  std::string type;           // feature type (lake, river, beach, etc.)
  glm::vec3 color;            // color for rendering
};

// tree mesh data
struct tree_mesh {
  glm::vec3 position;         // position in 3D space
  float scale;                // size scaling factor
  float rotation;             // y-axis rotation in radians
  int type;                   // 0 = conifer, 1 = deciduous
};

// car mesh data for traffic simulation
struct car_mesh {
  glm::vec3 position;         // position in 3D space
  float rotation;             // y-axis rotation in radians
  int type;                   // 0 = sedan, 1 = SUV, etc.
};

// rain particle for weather effects
struct rain_particle {
  glm::vec3 position;         // position in 3D space
  glm::vec3 velocity;         // movement vector
  float age;                  // current lifetime
  float life;                 // total lifetime
};

//==============================================================================
// MATERIAL AND VISUAL PROPERTIES
//==============================================================================
// material properties for Phong lighting model
struct material_properties {
  float ambient;              // ambient reflection coefficient
  float diffuse;              // diffuse reflection coefficient
  float specular;             // specular reflection coefficient
  float shininess;            // specular exponent
};

// water properties for realistic water rendering
struct water_properties {
  float wave_strength;        // amplitude of waves
  float wave_speed;           // speed of wave movement
  float wave_frequency;       // frequency of waves
  float reflectivity;         // surface reflectivity factor
  float refraction_strength;  // refraction intensity
  float specular_power;       // specular highlight intensity
  glm::vec3 color;            // base water color
};


//==============================================================================
// MAIN RENDERER CLASS
//==============================================================================
// Main 3D rendering class handling all visualization aspects
class Renderer3D {
public:
    // Constructor initializes the renderer with a starting position
    Renderer3D(const latlon& initial_pos);

    // Destructor cleans up all resources
    ~Renderer3D();

    // Main rendering loop
    void main_loop();

    //==========================================================================
    // INTERNAL HELPER FUNCTIONS
    //==========================================================================
    glm::vec3 latlon_to_3d(const latlon& coord, const latlon& reference);
    // converts color structure to glm::vec3 for shader input
    glm::vec3 color_to_vec3(const color& c);
    // sets the rendering distance radius - controls how far objects are visible
    void set_render_radius(float radius) { render_radius = radius; }
    // gets the current rendering distance radius
    float get_render_radius() const { return render_radius; }
    // enhances rain particle effects - adds more particles and improves visuals
    void enhance_rain_particles();


private:
    //==========================================================================
    // OPENGL/GLFW SHADER PROGRAMS
    //==========================================================================
    GLFWwindow* window; // main rendering window - application window
    // gluint handles gpu programs that process and transform data
    GLuint shader_program; // basic shader - minimal effects
    GLuint phong_shader_program; // phong lighting shader
    GLuint tess_shader_program; // tessellation shader
    GLuint water_shader_program; // water shader
    GLuint shadow_shader_program; // shadow mapping shader
    GLuint terrain_shader_program; // terrain shader
    GLuint particle_shader_program; // particle system shader
    

    //==========================================================================
    // OPENGL BUFFERS
    //==========================================================================
    // vao is vertex attribute data
    GLuint building_vao; // vertex array object for buildings - stores config for building geometry
    GLuint vao, vbo; // main vertex array and buffer - general purpose vertex objects

    //==========================================================================
    // TERRAIN VARAIBLES
    //==========================================================================
    GLuint terrain_vao; // terrain vertex array - terrain mesh rendering
    GLuint terrain_vbo_position; // terrain vertex positions - 3d coordinates of vertices
    GLuint terrain_vbo_tex_coords; // terrain texture coordinates - normal vectors for lighting calculations - tex = texture
    GLuint terrain_ebo; // terrain element buffer - element buffer object
    std::vector<float> terrain_vertices; // terrain vertex data
    std::vector<float> terrain_tex_coords; // terrain texture coordinates
    std::vector<unsigned int> terrain_indices; // terrain index data
    float terrain_noise_scale; // controls scale of terrain noise
    float terrain_noise_height; // controls height of terrain features
    glm::vec2 terrain_noise_offset; //offset for noise sampling


    //==========================================================================
    // RENDERING PARAMETERS
    //==========================================================================
    GLuint building_vbo_position; // building position buffer
    GLuint building_vbo_normal; // building normal buffer
    GLuint building_vbo_texcoord; // building texture coordinate buffer



    //==========================================================================
    // CAMERA + DEVICE VARIABLES
    //==========================================================================
    glm::vec3 camera_pos; // camera position
    glm::vec3 camera_front; // camera look direction
    glm::vec3 camera_up; // camera up vector
    float yaw, pitch; // camera orientation angles
    double last_x, last_y; // last mouse position
    bool first_mouse; // first mouse movement flag
    latlon reference_point; // reference point for coordinate conversion
    float last_frame; // time of last frame for timing
    float last_fps_time; // time of last FPS calculation
    float render_radius; // maximum render distance in meters


    //==========================================================================
    // LIGHTING
    //==========================================================================
    glm::vec3 light_position; // sun/main light position
    glm::vec3 light_color; // light color
    glm::mat4 light_space_matrix; // matrix for light-space transforms
    

    //==========================================================================
    // SKY
    //==========================================================================
    GLuint sky_shader;                  // shader for skybox rendering
    GLuint sky_vao;                     // vertex array for sky
    GLuint sky_vbo;                     // vertex buffer for sky
    void init_sky();                    // initializes sky rendering resources
    void render_sky();                  // renders the skybox
    

    //==========================================================================
    // MATERIALS
    //==========================================================================
    // pre-defined material properties for different object types
    // defines how light interacts with different surfaces
    material_properties building_material;
    material_properties road_material;
    material_properties water_material;
    material_properties terrain_material;
    
    
    //==========================================================================
    // WATER AND RAIN EFFECTS
    //==========================================================================
    // weather and effects parameters
    float rain_intensity; // controls amount of rain
    int max_rain_particles; // maximum number of rain particles
    float particle_size; // size of particles
    float tessellation_level; // level of detail for tessellation
    // water rendering properties
    water_properties water_props; // water visual properties

    //==========================================================================
    // SHADOW MAPPING RESOURCES
    //==========================================================================
    GLuint shadow_map_fbo;              // framebuffer for shadow rendering
    GLuint shadow_map_texture;          // shadow map texture
    
    //==========================================================================
    // MAIN VISUAL DATA
    //==========================================================================
    std::vector<building> buildings;
    std::vector<road_segment> road_segments;
    std::vector<natural_feature> natural_features;
    std::vector<tree_mesh> trees;
    std::vector<car_mesh> cars;
    std::vector<rain_particle> rain_particles;
    
    //==========================================================================
    // TEXTURE DATA
    //==========================================================================
    // handles 3d references from images
    GLuint grass_texture;
    GLuint rock_texture;
    GLuint soil_texture;
    GLuint concrete_texture;
    GLuint window_texture;
    GLuint road_texture;
    GLuint tree_texture;
    GLuint car_texture;

    //==========================================================================
    // THREAD DATA
    //==========================================================================
    std::mutex map_data_mutex; // mutex for thread-safe data access
    // could potentially remove this
    
    //==========================================================================
    // CUDA DATA
    //==========================================================================
    // building data on GPU
    gpu_building_data* d_building_data = nullptr; // device pointer to building data
    float3* d_building_vertices = nullptr; // device pointer to building vertices
    int building_count = 0; // number of buildings
    
    // road data on GPU
    gpu_road_data* d_road_data = nullptr; // device pointer to road data
    float3* d_road_vertices = nullptr; // device pointer to road vertices
    int road_count = 0; // number of roads
    
    // natural feature data on GPU
    gpu_natural_feature_data* d_natural_feature_data = nullptr; // device pointer to natural feature data
    float3* d_natural_feature_vertices = nullptr; // device pointer to natural feature vertices
    int natural_feature_count = 0; // number of natural features
    
    // visibility data for culling
    int* d_visible_building_indices = nullptr; // indices of visible buildings
    int* d_visible_building_count = nullptr; // number of visible buildings
    int visible_building_count = 0; // host copy of visible building count
    
    int* d_visible_road_indices = nullptr; // indices of visible roads
    int* d_visible_road_count = nullptr; // number of visible roads
    int visible_road_count = 0; // host copy of visible road count
    
    int* d_visible_natural_feature_indices = nullptr; // indices of visible natural features
    int* d_visible_natural_feature_count = nullptr; // number of visible natural features
    int visible_natural_feature_count = 0; // host copy of visible natural feature count
    
    // frustum culling data
    float4* d_frustum_planes = nullptr; // frustum planes for culling
    
    // spatial grid for acceleration
    cuda_grid d_grid; // spatial grid on device

    //==========================================================================
    // INITIALIZATION METHODS
    //==========================================================================
    // sets up basic shader programs
    void setup_shaders();
    
    // sets up phong lighting shader programs
    void setup_phong_shaders();
    
    // sets up tessellation shader programs
    void setup_tessellation_shaders();
    
    // sets up water shader programs
    void setup_water_shaders();
    
    // sets up shadow mapping resources
    void setup_shadow_mapping();
    
    // sets up terrain shader programs
    void setup_terrain_shaders();
    
    // sets up particle system resources
    void setup_particle_system();
    
    // initializes CUDA resources
    void init_cuda_resources();
    
    // initializes spatial grid for efficient culling
    // divides the world into a grid for faster object lookup
    void init_spatial_grid();
    
    // initializes rain particle system
    void init_rain_particles();
    
    // generates terrain mesh based on noise functions
    void generate_terrain();
    
    // generates tree placements based on terrain features
    void generate_trees();
    
    // generates car placements on roads
    void generate_cars();
    
    // loads texture resources from files
    void load_textures();

    //==========================================================================
    // DATA LOADING METHODS
    //==========================================================================
    // loads map features from database
    void load_features();
    
    // loads road data from database
    void load_roads();
    
    // loads natural features from database
    void load_natural_features();
    
    // uploads building data to GPU for faster processing
    void upload_buildings_to_gpu();
    
    // uploads road data to GPU for faster processing
    void upload_roads_to_gpu();
    
    // uploads natural feature data to GPU for faster processing
    void upload_natural_features_to_gpu();

    //==========================================================================
    // INPUT HANDLING METHODS
    //==========================================================================
    // processes keyboard input for camera movement
    void process_input(float delta_time);
    
    // handles mouse movement for camera control
    void mouse_callback(double xpos, double ypos);

    //==========================================================================
    // RENDERING & DRAW METHODS
    //==========================================================================
    // renders shadow map for shadow calculation
    // creates depth map from light's perspective for shadow rendering
    void render_shadow_map();
    
    // renders terrain mesh
    void render_terrain();
    
    // draws bright terrain with sun reflection
    // enhances terrain with specular highlights for sun reflection
    void render_bright_terrain();
    
    // draws all buildings in the scene
    void draw_buildings();
    
    // draws building walls with proper texturing
    void draw_building_walls(const building& building);
    
    // draws building roof with appropriate style
    void draw_building_roof(const building& building);

    // draws a simple flat roof for a building
    void draw_simple_flat_roof(const building& building);
    
    // draws building antenna (if applicable)
    // adds detail to tall buildings
    void draw_building_antenna(const building& building);
    
    // draws all visible roads
    void draw_roads();
    
    // draws visible single road segment
    void draw_road_segment(const road_segment& road);
    
    // draws visible natural features (lakes, parks, etc.)
    void draw_natural_features();
    
    // draws all trees in the scene
    void draw_trees();
    
    // draws a single tree mesh
    void draw_tree_mesh(const tree_mesh& tree);
    // draws the trunk part of a tree
    void draw_tree_trunk(const tree_mesh& tree);
    // draws the foliage part of a tree
    void draw_tree_foliage(const tree_mesh& tree);
    // draws pine tree foliage (conical shape)
    void draw_pine_tree_foliage(const tree_mesh& tree);
    
    // draws all cars in the scene
    void draw_cars();
    
    // renders rain particles for weather effects
    void render_rain_particles();
    
    // renders darker green areas for parks
    // makes park areas visually distinct from other terrain
    void render_darker_greenspace(const natural_feature& feature);
    
    // adds trees to green spaces for visual enhancement
    void add_trees_to_greenspace(const natural_feature& feature);
    
    // draws terrain using standard approach
    void draw_terrain();

    //==========================================================================
    // RENDERING METHODS
    //==========================================================================
    // draws mesh with phong lighting model
    void draw_with_phong(const std::vector<float>& vertices, 
                        const std::vector<float>& normals,
                        const glm::vec3& color, 
                        const material_properties& material);
    
    // draws mesh with tessellation for adaptive level of detail
    void draw_with_tessellation(const std::vector<float>& vertices, 
                               const std::vector<float>& normals,
                               const glm::vec3& color, 
                               const material_properties& material,
                               float tess_level);
    
    // draws water surface with specialized shader
    // renders realistic water with reflection, refraction, and waves
    void draw_water_surface(const std::vector<float>& vertices, 
                           const std::vector<float>& tex_coords,
                           const water_properties& water_props);
    
    // draws water feature with specialized water shader
    void draw_water_feature(const natural_feature& feature);
    
    // draws terrain with perlin noise for natural variations
    void draw_terrain_with_perlin(const std::vector<float>& vertices,
                                 const std::vector<float>& tex_coords);

    //==========================================================================
    // UTILITY METHODS
    //==========================================================================
    // calculates normals for a mesh based on vertex positions
    std::vector<float> calculate_normals(const std::vector<float>& vertices);
   
    // generates a grid of given size and resolution
    // creates a 2D grid of vertices for terrain or water
    std::vector<float> generate_grid(float size, int resolution);
    
    // generates texture coordinates for a grid
    std::vector<float> generate_grid_tex_coords(int resolution);
    
    // updates rain particle positions based on physics
    void update_rain_particles(float delta_time);
    
    // create primitive shapes for composite objects
    // creates a sphere shape for object components
    void create_sphere_shape(std::vector<float>& vertices, std::vector<float>& normals, 
                          float center_y, float radius);
    // creates a cone shape for object components
    void create_cone_shape(std::vector<float>& vertices, std::vector<float>& normals, 
                        float base_height, float cone_height, float radius);
    // adds a cylinder mesh to an existing vertex array
    void add_cylinder_mesh(std::vector<float>& vertices, std::vector<float>& normals, 
                           const glm::vec3& center, float radius, float width, int segments);
    
    //==========================================================================
    // FRUSTUM CULLING METHODS
    //==========================================================================
    // computes frustum planes for culling
    // extracts viewing frustum planes from view-projection matrix
    void compute_frustum_planes(glm::vec4* planes);
    
    // performs frustum culling on scene objects
    // determines which objects are visible in the current view
    void perform_frustum_culling();
    
    //==========================================================================
    // CUDA UTILITY METHODS
    //==========================================================================
    // checks CUDA errors and reports issues
    void cuda_check(cudaError_t err, const char* context);
    
    // cleans up CUDA resources to prevent memory leaks
    void cleanup_cuda_resources();
    
    // cleans up natural feature resources
    void cleanup_natural_features();

    //==========================================================================
    // SHADER UTILITIES
    //==========================================================================
    // compiles a shader from source code
    GLuint compile_shader(GLenum type, const char* source);
    
    // creates a shader program from vertex and fragment shaders
    GLuint create_shader_program(const char* vertex_source, const char* fragment_source);
    
    // creates a shader program with tessellation stages
    // includes tessellation control and evaluation shaders for LOD control
    GLuint create_shader_program_with_tessellation(const char* vertex_source, 
                                                 const char* tess_control_source,
                                                 const char* tess_eval_source,
                                                 const char* fragment_source);
    
    // shader sources as static constants
    static const char* vertex_shader;
    static const char* fragment_shader;
};

#endif // _3D_RENDERING_H_