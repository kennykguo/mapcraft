#include <3d_rendering.h>
#include "3d_tesselation.cpp"
#include "3d_particles.cpp"
#include "3d_phong_lighting.cpp"
#include "3d_terrain.cpp"
#include "3d_shadows.cpp"
#include "3d_water.cpp"
#include "3d_textures.cpp"
#include "3d_renderer3d.cpp"
#include "3d_shaders.cpp"
#include "3d_load.cpp"
#include "3d_cuda_init_load.cpp"
#include "3d_frustum_culling.cpp"
#include "3d_trees.cpp"
#include "3d_helpers.cpp"
#include "3d_cars.cpp"
#include "3d_draw.cpp"
#include "3d_sky.cpp"
#include "serialization.cpp"


// global data structures
std::vector<street_segment_data> graphics_street_segment_data;
std::vector<feature_data> graphics_feature_data;
latlon reference_point;
const double FPS_PRINT = 5.0f;
const glm::vec3 SUN_POSITION(500.0f, 1000.0f, 500.0f); // fixed sun position
const glm::vec3 SUN_COLOR(1.0f, 1.0f, 0.9f);  

void Renderer3D::main_loop() {
    // disable cursor for camera control and set up mouse callback
    // window is a glfw window handle, GLFW_CURSOR specifies we're changing cursor behaviour, and GLFW_CURSOR_DISABLED hides the cursor
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);// hides the mouse cursor, and captures it within the window for FPS
    glfwSetWindowUserPointer(window, this); // stores a pointer to the Renderer3D object with the window for callbacks
    glfwSetCursorPosCallback(window, [](GLFWwindow* w, double x, double y) {
        static_cast<Renderer3D*>(glfwGetWindowUserPointer(w))->mouse_callback(x, y);
    }); // registers the callback function to handle mouse movement for camera control -> executes whenever the mouse moves
    
    // scene elements
    generate_terrain(); // create terrain mesh
    load_textures(); // load texture assets
    generate_trees(); // place trees in the world
    generate_cars(); // place cars on roads
    init_rain_particles(); // initialize rain particle system
    init_sky(); // initialize sky
    
    // fixed sun position for consistent lighting
    light_position = SUN_POSITION;
    light_color = SUN_COLOR;
    
    // main rendering loop - continues until window is closed
    while (!glfwWindowShouldClose(window)) {
        // frame timing logic
        float current_time = glfwGetTime();
        float delta_time = current_time - last_frame;
        last_frame = current_time;
        
        process_input(delta_time); // handle keyboard/mouse input
        update_rain_particles(delta_time); // update dynamic elements
        perform_frustum_culling(); // frustum culling on gpu
        render_shadow_map();  // first render pass: shadow map from light's perspective
        
        // second rendering pass: render the scene with shadows
        // clear only depth and stencil buffers, preserving color buffer for sky
        // glClear erases specific buffers to preset values
        // GL_DEPTH_BUFFER_BIT clears the depth buffer to the far clipping plane, GL_STENCIL_BUFFER_BIT clears the stencil buffer to 0 
        // depth buffer is reset to the maximum (1.0), and stencil is reset to zero
        // stencil buffer stores int value for each pixel
        glClear(GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT); 
        
        // draw skybox as first element (no depth testing)
        render_sky(); 

        // enable depth testing for proper object ordering
        glEnable(GL_DEPTH_TEST); // enables depth testing
        glDepthFunc(GL_LESS); // sets the comparison to less than - a pixel is drawn only if its depth value is less than the existing value
        // in simple words, we only draw a pixel on the screen if there is some object with a closer depth (how far away from the camera), than hte vlaue stored in the dpeth buffer)
        // render scene elements in proper order for correct depth handling:

        // 1. terrain forms the base layer
        render_terrain();
        
        // 2. roads are drawn slightly above terrain to avoid "z-fighting"
        glEnable(GL_POLYGON_OFFSET_FILL); // activates polygon offset
        glPolygonOffset(-1.0f, -1.0f); // negative offset brings roads above terrain
        // roads appear slightly infront of terrain - 1. scale factor, 2. constant bias to every fragment
        draw_roads();
        glDisable(GL_POLYGON_OFFSET_FILL);
        
        // 3. natural features
        draw_natural_features();
        
        // 4. buildings with proper stencil buffer setup for windows/details
        // a stencil is like a layer in opengl
        // mark every pixel for buildings drawn with a 1, so that twe can draw windows, lighting, reflections, etc
        glEnable(GL_STENCIL_TEST);
        glStencilFunc(GL_ALWAYS, 1, 0xFF); // always passes test, 1 is the ref value to compare, and 0XFF is a mask
        // determines if we need to redraw the building or not -> only when we have a new pixel that passes both tests
        glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE); // 1. stencil test fails, 2. stencil pass, but depth fails, 3. both tests pass
        glStencilMask(0xFF);
        draw_buildings();
        glStencilMask(0x00); // disable writing to stencil buffer after buildings
        
        
        // 5. cars are rendered on top of roads
        draw_cars();
        
        // 6. trees are drawn in parks and along roads
        draw_trees();
        
        glDisable(GL_STENCIL_TEST); // clean up stencil state
        // 7. rain particles
        // render_rain_particles();
        // enhance_rain_particles();
        
        // swap buffers to display the rendered frame (front and back buffer logic in the bg)
        glfwSwapBuffers(window);
        
        // process window events (resize, keyboard, etc.)
        glfwPollEvents();
        
        // display debug information once per second
        if (current_time - last_fps_time >= FPS_PRINT) {
            float fps = FPS_PRINT / delta_time;
            std::cout << "FPS: " << fps 
                      << " (roads: " << visible_road_count 
                      << ", buildings: " << visible_building_count 
                      << ", features: " << visible_natural_feature_count 
                      << ", trees: " << trees.size()
                      << ", cars: " << cars.size() << ")" << std::endl;
            last_fps_time = current_time;
        }
    }
}

void start_3d_view(const latlon& position) {
    try {
        std::cout<< "initializing renderer" << std::endl;
        Renderer3D renderer(position);
        renderer.main_loop();
    }
    catch (const std::exception& e) {
        std::cerr << "3D view error: " << e.what() << std::endl;
    }
}

void start_3d_rendering(double latitude, double longitude) {
  latlon ref_latlon = latlon(latitude, longitude);
  std::cout << "reference point initialized at: lat=" << ref_latlon.latitude << ", lon=" << ref_latlon.longitude << std::endl;
  // loads and processes the serialized .bin data
  load_serialized_data(ref_latlon);
  // functionality defined above
  start_3d_view(ref_latlon);
  return;
}