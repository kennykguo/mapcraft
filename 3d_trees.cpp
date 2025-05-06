#include "3d_rendering.h"
// tree generation constants - these control how dense and varied our forests appear
const float TREES_PER_SQ_METER = 1.0f / 75.0f; // one tree per ~75 square meters
const float MIN_TREES_PER_AREA = 5.0f; // minimum trees even in small spaces  
const float TREE_SCALE_MIN = 0.7f; // smallest tree scale
const float TREE_SCALE_MAX = 5.0f; // largest tree scale
const float TRUNK_HEIGHT = 8.0f; // standard pine trunk height
const float TRUNK_RADIUS = 0.3f; // trunk thickness
const float CONE_HEIGHT = 12.0f; // pine cone foliage height
const float CONE_RADIUS = 4.0f; // pine cone base width
const float ROAD_TREE_SPACING = 20.0f; // meters between road trees
const float ROAD_TREE_CHANCE = 0.7f; // 70% chance to place each tree

void Renderer3D::generate_trees() {
    // clear previous trees - prevents memory leaks and duplicate rendering
    std::cout << "generating tree meshes..." << std::endl;trees.clear();
    std::random_device rd;std::mt19937 gen(rd());
    
    // first stage: process greenspaces and parks - these are natural tree habitats
    std::vector<natural_feature*> greenspaces;
    // looping over all natural features (not yet culled)
    for (auto& feature : natural_features) {
        if (feature.type == "greenspace" || feature.type == "park") {greenspaces.push_back(&feature);}
    }
    
    // second stage: populate greenspaces with trees using natural distribution
    for (natural_feature* space : greenspaces) {
        // calculate area using triangulation method - divides polygon into triangles from centroid
        float area = 0;glm::vec3 centroid = space->centroid;
        
        // for each segment of the greenspace boundary, calculate area of triangle with centroid
        for (int i = 0; i < space->vertex_count; i++) {
            int next_i = (i + 1) % space->vertex_count;
            glm::vec3 v1 = space->vertices[i];glm::vec3 v2 = space->vertices[next_i];
            // calculate area of this triangle using cross product for area calculation
            glm::vec3 cross_product = glm::cross(v1 - centroid, v2 - centroid);
            area += 0.5f * glm::length(cross_product);}
        
        // determine tree density based on area - more trees in larger spaces
        int num_trees = std::max(static_cast<int>(MIN_TREES_PER_AREA), static_cast<int>(area * TREES_PER_SQ_METER));
        
        // prepare random distributions for natural variety
        std::uniform_real_distribution<float> scale_dist(TREE_SCALE_MIN, TREE_SCALE_MAX);
        std::uniform_real_distribution<float> rot_dist(0.0f, 360.0f);
        std::uniform_int_distribution<int> type_dist(0, 1);
        
        // place trees using barycentric coordinates for even distribution
        for (int i = 0; i < num_trees; i++) {
            tree_mesh tree;
            
            // random position within the greenspace using barycentric coordinates
            std::vector<float> weights(space->vertex_count);float total_weight = 0;
            
            // generate random weights for each vertex - creates natural clustering
            for (int j = 0; j < space->vertex_count; j++) {
                weights[j] = std::uniform_real_distribution<float>(0.0f, 1.0f)(gen);total_weight += weights[j];}
            
            // normalize weights to sum to 1 - ensures position is inside polygon
            for (int j = 0; j < space->vertex_count; j++) {weights[j] /= total_weight;}
            
            // compute weighted position - interpolates between polygon vertices
            glm::vec3 position(0.0f);
            for (int j = 0; j < space->vertex_count; j++) {position += weights[j] * space->vertices[j];}
            
            // add small random offset for natural appearance - prevents grid-like patterns  
            position.x += std::uniform_real_distribution<float>(-10.0f, 10.0f)(gen);
            position.z += std::uniform_real_distribution<float>(-10.0f, 10.0f)(gen);
            position.y = space->elevation + 0.5f; // slightly above ground to prevent z-fighting
            
            tree.position = position;tree.scale = scale_dist(gen);tree.rotation = rot_dist(gen);tree.type = type_dist(gen);
            trees.push_back(tree);}
    }
    
    // third stage: add trees along residential roads for urban aesthetics
    for (const auto& road : road_segments) {
        // only add trees along residential roads (type 3) - highways don't have trees
        if (road.road_type != 3) continue;if (road.vertex_count < 2) continue;
        
        // prepare randomization for road trees
        std::uniform_real_distribution<float> scale_dist(TREE_SCALE_MIN, TREE_SCALE_MAX);
        std::uniform_real_distribution<float> rot_dist(0.0f, 360.0f);
        std::uniform_int_distribution<int> type_dist(0, 1);
        std::uniform_real_distribution<float> skip_dist(0.0f, 1.0f);
        
        // process each road segment individually
        for (int i = 0; i < road.vertex_count - 1; i++) {
            glm::vec3 p1 = road.vertices[i];glm::vec3 p2 = road.vertices[i + 1];
            
            // calculate direction and perpendicular vectors for tree placement
            glm::vec3 dir = glm::normalize(p2 - p1);glm::vec3 perp = glm::normalize(glm::cross(dir, glm::vec3(0, 1, 0)));
            float segment_length = glm::distance(glm::vec2(p1.x, p1.z), glm::vec2(p2.x, p2.z));
            int num_trees = static_cast<int>(segment_length / ROAD_TREE_SPACING);
            
            for (int j = 0; j < num_trees; j++) {
                // random skip for natural gaps between trees
                if (skip_dist(gen) > ROAD_TREE_CHANCE) continue;
                
                // position along the segment - evenly spaced with offset
                float t = (j + 0.5f) / num_trees;glm::vec3 pos = p1 + dir * (t * segment_length);
                
                // offset to the sides of the road with random variation
                float side_offset = road.width * 0.7f;float side_random = std::uniform_real_distribution<float>(-2.0f, 2.0f)(gen);
                
                // randomly choose left or right side of road
                int side = std::uniform_int_distribution<int>(0, 1)(gen);
                if (side == 0) {pos += perp * (side_offset + side_random);} 
                else {pos -= perp * (side_offset + side_random);}
                
                pos.y = 0.2f; // slightly above ground level
                
                tree_mesh tree;tree.position = pos;tree.scale = scale_dist(gen);tree.rotation = rot_dist(gen);tree.type = type_dist(gen);
                trees.push_back(tree);}
        }
    }
    
    std::cout << "generated " << trees.size() << " trees" << std::endl;
}

void Renderer3D::draw_tree_mesh(const tree_mesh& tree) {
    // create model matrix for this tree - transforms from local to world space
    glm::mat4 model = glm::mat4(1.0f);
    model = glm::translate(model, tree.position); // move to position
    model = glm::rotate(model, glm::radians(tree.rotation), glm::vec3(0.0f, 1.0f, 0.0f)); // rotate
    model = glm::scale(model, glm::vec3(tree.scale)); // scale to desired size
    
    glUniformMatrix4fv(glGetUniformLocation(phong_shader_program, "model"), 1, GL_FALSE, glm::value_ptr(model));
    
    draw_tree_trunk(tree); // brown cylinder for structure
    draw_pine_tree_foliage(tree); // green cone for leaves
}

void Renderer3D::draw_tree_trunk(const tree_mesh& tree) {
    // set brown color for trunk - natural wood appearance
    glm::vec3 trunk_color(0.55f, 0.35f, 0.15f);
    glUniform3fv(glGetUniformLocation(phong_shader_program, "objectColor"), 1, glm::value_ptr(trunk_color));
    
    // create cylinder for trunk - efficient representation of round objects
    std::vector<float> vertices;std::vector<float> normals;int segments = 8;
    
    // create cylinder vertices using parametric equations
    for (int i = 0; i < segments; i++) {
        float angle1 = glm::radians(static_cast<float>(i) / segments * 360.0f);
        float angle2 = glm::radians(static_cast<float>(i + 1) / segments * 360.0f);
        
        // calculate vertices for this segment
        float x1 = cos(angle1) * TRUNK_RADIUS;float z1 = sin(angle1) * TRUNK_RADIUS;
        float x2 = cos(angle2) * TRUNK_RADIUS;float z2 = sin(angle2) * TRUNK_RADIUS;
        
        // add triangles for bottom face
        vertices.push_back(0.0f); vertices.push_back(0.0f); vertices.push_back(0.0f);
        vertices.push_back(x1); vertices.push_back(0.0f); vertices.push_back(z1);
        vertices.push_back(x2); vertices.push_back(0.0f); vertices.push_back(z2);
        
        // add downward-pointing normals for bottom
        normals.push_back(0.0f); normals.push_back(-1.0f); normals.push_back(0.0f);
        normals.push_back(0.0f); normals.push_back(-1.0f); normals.push_back(0.0f);
        normals.push_back(0.0f); normals.push_back(-1.0f); normals.push_back(0.0f);
        
        // add triangles for top face
        vertices.push_back(0.0f); vertices.push_back(TRUNK_HEIGHT); vertices.push_back(0.0f);
        vertices.push_back(x2); vertices.push_back(TRUNK_HEIGHT); vertices.push_back(z2);
        vertices.push_back(x1); vertices.push_back(TRUNK_HEIGHT); vertices.push_back(z1);
        
        // add upward-pointing normals for top
        normals.push_back(0.0f); normals.push_back(1.0f); normals.push_back(0.0f);
        normals.push_back(0.0f); normals.push_back(1.0f); normals.push_back(0.0f);
        normals.push_back(0.0f); normals.push_back(1.0f); normals.push_back(0.0f);
        
        // add triangles for side faces
        vertices.push_back(x1); vertices.push_back(0.0f); vertices.push_back(z1);
        vertices.push_back(x1); vertices.push_back(TRUNK_HEIGHT); vertices.push_back(z1);
        vertices.push_back(x2); vertices.push_back(TRUNK_HEIGHT); vertices.push_back(z2);
        
        // calculate outward-pointing normals for cylinder surface
        glm::vec3 side_normal1 = glm::normalize(glm::vec3(x1, 0.0f, z1));
        glm::vec3 side_normal2 = glm::normalize(glm::vec3(x2, 0.0f, z2));
        normals.push_back(side_normal1.x); normals.push_back(0.0f); normals.push_back(side_normal1.z);
        normals.push_back(side_normal1.x); normals.push_back(0.0f); normals.push_back(side_normal1.z);
        normals.push_back(side_normal2.x); normals.push_back(0.0f); normals.push_back(side_normal2.z);
        
        vertices.push_back(x1); vertices.push_back(0.0f); vertices.push_back(z1);
        vertices.push_back(x2); vertices.push_back(TRUNK_HEIGHT); vertices.push_back(z2);
        vertices.push_back(x2); vertices.push_back(0.0f); vertices.push_back(z2);
        
        normals.push_back(side_normal1.x); normals.push_back(0.0f); normals.push_back(side_normal1.z);
        normals.push_back(side_normal2.x); normals.push_back(0.0f); normals.push_back(side_normal2.z);
        normals.push_back(side_normal2.x); normals.push_back(0.0f); normals.push_back(side_normal2.z);}
    
    // create opengl resources for rendering
    GLuint trunk_vao, vbo_position, vbo_normal;
    glGenVertexArrays(1, &trunk_vao);glGenBuffers(1, &vbo_position);glGenBuffers(1, &vbo_normal);
    glBindVertexArray(trunk_vao);
    
    // upload vertex data to gpu
    glBindBuffer(GL_ARRAY_BUFFER, vbo_position);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    // upload normal data to gpu
    glBindBuffer(GL_ARRAY_BUFFER, vbo_normal);
    glBufferData(GL_ARRAY_BUFFER, normals.size() * sizeof(float), normals.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    
    glDrawArrays(GL_TRIANGLES, 0, vertices.size() / 3);
    
    // cleanup - prevent resource leaks
    glDeleteVertexArrays(1, &trunk_vao);glDeleteBuffers(1, &vbo_position);glDeleteBuffers(1, &vbo_normal);
}

void Renderer3D::draw_pine_tree_foliage(const tree_mesh& tree) {
    // dark green color for pine foliage - evergreen appearance
    glm::vec3 foliage_color = glm::vec3(0.1f, 0.4f, 0.1f);
    glUniform3fv(glGetUniformLocation(phong_shader_program, "objectColor"), 1, glm::value_ptr(foliage_color));
    
    // create geometry for pine cone shape
    std::vector<float> vertices;std::vector<float> normals;
    
    // create the pine tree cone shape - classic christmas tree appearance
    create_cone_shape(vertices, normals, TRUNK_HEIGHT, CONE_HEIGHT, CONE_RADIUS);
    
    // create vao and vbos for the foliage
    GLuint foliage_vao, vbo_position, vbo_normal;
    glGenVertexArrays(1, &foliage_vao);glGenBuffers(1, &vbo_position);glGenBuffers(1, &vbo_normal);
    glBindVertexArray(foliage_vao);
    
    // upload position data
    glBindBuffer(GL_ARRAY_BUFFER, vbo_position);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    // upload normal data
    glBindBuffer(GL_ARRAY_BUFFER, vbo_normal);
    glBufferData(GL_ARRAY_BUFFER, normals.size() * sizeof(float), normals.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    
    glDrawArrays(GL_TRIANGLES, 0, vertices.size() / 3);
    
    // cleanup
    glDeleteVertexArrays(1, &foliage_vao);glDeleteBuffers(1, &vbo_position);glDeleteBuffers(1, &vbo_normal);
}

void Renderer3D::create_cone_shape(std::vector<float>& vertices, std::vector<float>& normals,  float baseHeight, float coneHeight, float radius) {
    int segments = 16; // number of segments around circumference - higher = smoother
    float base_y = baseHeight;float tip_y = baseHeight + coneHeight;
    
    // create cone vertices using parametric equations
    for (int i = 0; i < segments; i++) {
        float angle1 = glm::radians(static_cast<float>(i) / segments * 360.0f);
        float angle2 = glm::radians(static_cast<float>(i + 1) / segments * 360.0f);
        
        // calculate base vertices
        float x1 = cos(angle1) * radius;float z1 = sin(angle1) * radius;
        float x2 = cos(angle2) * radius;float z2 = sin(angle2) * radius;
        
        // add bottom face triangle
        vertices.push_back(0.0f); vertices.push_back(base_y); vertices.push_back(0.0f);
        vertices.push_back(x2); vertices.push_back(base_y); vertices.push_back(z2);
        vertices.push_back(x1); vertices.push_back(base_y); vertices.push_back(z1);
        
        // downward-pointing normals for base
        normals.push_back(0.0f); normals.push_back(-1.0f); normals.push_back(0.0f);
        normals.push_back(0.0f); normals.push_back(-1.0f); normals.push_back(0.0f);
        normals.push_back(0.0f); normals.push_back(-1.0f); normals.push_back(0.0f);
        
        // add side triangle
        vertices.push_back(x1); vertices.push_back(base_y); vertices.push_back(z1);
        vertices.push_back(x2); vertices.push_back(base_y); vertices.push_back(z2);
        vertices.push_back(0.0f); vertices.push_back(tip_y); vertices.push_back(0.0f);
        
        // calculate normals for cone side - requires cross product calculation
        glm::vec3 side1(x1, base_y, z1);glm::vec3 side2(x2, base_y, z2);glm::vec3 tip(0.0f, tip_y, 0.0f);
        glm::vec3 edge1 = side2 - side1;glm::vec3 edge2 = tip - side1;
        glm::vec3 normal = glm::normalize(glm::cross(edge1, edge2));
        
        normals.push_back(normal.x); normals.push_back(normal.y); normals.push_back(normal.z);
        normals.push_back(normal.x); normals.push_back(normal.y); normals.push_back(normal.z);
        normals.push_back(normal.x); normals.push_back(normal.y); normals.push_back(normal.z);}
}

void Renderer3D::draw_trees() {
    // use phong shader for realistic lighting on trees
    if (trees.empty()) return;glUseProgram(phong_shader_program);
    
    // set transformation matrices for camera and perspective
    glm::mat4 view = glm::lookAt(camera_pos, camera_pos + camera_front, camera_up);
    glm::mat4 projection = glm::perspective(glm::radians(45.0f), 1024.0f/768.0f, 0.1f, 10000.0f);
    glUniformMatrix4fv(glGetUniformLocation(phong_shader_program, "view"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(phong_shader_program, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
    
    // configure lighting parameters
    glUniform3fv(glGetUniformLocation(phong_shader_program, "viewPos"), 1, glm::value_ptr(camera_pos));
    glUniform3fv(glGetUniformLocation(phong_shader_program, "lightPos"), 1, glm::value_ptr(light_position));
    glUniform3fv(glGetUniformLocation(phong_shader_program, "lightColor"), 1, glm::value_ptr(light_color));
    
    // disable texturing for mesh-based trees - use solid colors instead
    glUniform1i(glGetUniformLocation(phong_shader_program, "useTexture"), 0);
    glUniform1i(glGetUniformLocation(phong_shader_program, "useWindows"), 0);
    
    // set material properties for natural appearance
    glUniform1f(glGetUniformLocation(phong_shader_program, "ambient"), 0.3f); // soft ambient light
    glUniform1f(glGetUniformLocation(phong_shader_program, "diffuse"), 0.7f); // main shading
    glUniform1f(glGetUniformLocation(phong_shader_program, "specular"), 0.1f); // minimal shine
    glUniform1f(glGetUniformLocation(phong_shader_program, "shininess"), 16.0f); // rough surface
    
    glEnable(GL_DEPTH_TEST); // ensure proper depth ordering
    
    // render each tree with distance culling for performance
    for (const auto& tree : trees) {
        // calculate distance to camera (2d distance only, ignoring height)
        float dist_sq = 
            (tree.position.x - camera_pos.x) * (tree.position.x - camera_pos.x) +
            (tree.position.z - camera_pos.z) * (tree.position.z - camera_pos.z);
        
        if (dist_sq > 500000.0f) continue; // skip trees beyond 500 meters
        draw_tree_mesh(tree);}
}