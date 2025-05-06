#include "3d_rendering.h"

void Renderer3D::load_textures() {
    std::cout << "loading textures..." << std::endl;
    grass_texture = TextureManager::loadTexture("grass.jpg");
    rock_texture = TextureManager::loadTexture("rock.jpg");
    soil_texture = TextureManager::loadTexture("soil.jpg");
    concrete_texture = TextureManager::loadTexture("concrete.jpg");
    window_texture = TextureManager::loadTexture("window.jpg");
    road_texture = TextureManager::loadTexture("road.jpg");
    tree_texture = TextureManager::loadTexture("tree.jpg");
    car_texture = TextureManager::loadTexture("car.jpg");
    
    std::cout << "All textures loaded successfully" << std::endl;
}
