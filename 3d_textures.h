#ifndef _TEXTURE_MANAGER_H_
#define _TEXTURE_MANAGER_H_

#include <glad/glad.h>
#include <string>
#include <map>
#include <iostream>
#include <vector>

// Simple class for loading and managing textures
class TextureManager {
public:
    // Load a texture from a file
    static GLuint loadTexture(const std::string& path) {
        GLuint textureID;
        glGenTextures(1, &textureID);
        
        // For this implementation, we'll simulate texture loading
        // In a real implementation, you would use stb_image.h or another library
        
        // Bind and set texture parameters
        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        
        // Generate a procedural texture based on the path
        int width = 512, height = 512;
        std::vector<unsigned char> data(width * height * 4);
        
        if (path.find("grass") != std::string::npos) {
            generateGrassTexture(data, width, height);
        } 
        else if (path.find("rock") != std::string::npos) {
            generateRockTexture(data, width, height);
        }
        else if (path.find("soil") != std::string::npos) {
            generateSoilTexture(data, width, height);
        }
        else if (path.find("concrete") != std::string::npos) {
            generateConcreteTexture(data, width, height);
        }
        else if (path.find("window") != std::string::npos) {
            generateWindowTexture(data, width, height);
        }
        else if (path.find("road") != std::string::npos) {
            generateRoadTexture(data, width, height);
        }
        else if (path.find("tree") != std::string::npos) {
            generateTreeTexture(data, width, height);
        }
        else if (path.find("car") != std::string::npos) {
            generateCarTexture(data, width, height);
        }
        else {
            // Default checker pattern
            generateDefaultTexture(data, width, height);
        }
        
        // Load the procedural texture to GPU
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data.data());
        glGenerateMipmap(GL_TEXTURE_2D);
        
        std::cout << "Generated procedural texture for: " << path << std::endl;
        
        return textureID;
    }
    
private:
    // Generate a grass texture
    // This function replaces the original generateGrassTexture in the TextureManager class
    static void generateGrassTexture(std::vector<unsigned char>& data, int width, int height) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // Lighter green base color
                unsigned char r = 100 + (rand() % 30);  // Increased from 50 to 100
                unsigned char g = 180 + (rand() % 40);  // Increased from 100 to 180
                unsigned char b = 80 + (rand() % 30);   // Increased from 30 to 80
                
                // Add some noise for grass variation
                float noise = (float)(rand() % 100) / 100.0f;
                r = (unsigned char)(r * (0.9f + noise * 0.2f));
                g = (unsigned char)(g * (0.9f + noise * 0.2f));
                b = (unsigned char)(b * (0.9f + noise * 0.2f));
                
                // Add occasional brighter grass blade
                if (rand() % 100 < 15) {
                    g += 30;
                    r += 20;
                }
                
                // Add occasional flower or detail
                if (rand() % 100 < 3) {
                    // Yellow flower
                    r = 220;
                    g = 220;
                    b = 50;
                } else if (rand() % 100 < 2) {
                    // White flower
                    r = 240;
                    g = 240;
                    b = 240;
                }
                
                // Set the pixel
                int index = (y * width + x) * 4;
                data[index] = r;
                data[index + 1] = g;
                data[index + 2] = b;
                data[index + 3] = 255; // Alpha
            }
        }
    }
    
    // Generate a rock texture
    static void generateRockTexture(std::vector<unsigned char>& data, int width, int height) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // Base gray color
                unsigned char val = 100 + (rand() % 50);
                
                // Add some noise for rock variation
                float noise = (float)(rand() % 100) / 100.0f;
                val = (unsigned char)(val * (0.9f + noise * 0.2f));
                
                // Add occasional cracks
                if (rand() % 100 < 5) {
                    val -= 30;
                }
                
                // Set the pixel
                int index = (y * width + x) * 4;
                data[index] = val;
                data[index + 1] = val;
                data[index + 2] = val;
                data[index + 3] = 255; // Alpha
            }
        }
    }
    
    // Generate a soil texture
    static void generateSoilTexture(std::vector<unsigned char>& data, int width, int height) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // Base brown color
                unsigned char r = 100 + (rand() % 50);
                unsigned char g = 70 + (rand() % 30);
                unsigned char b = 30 + (rand() % 20);
                
                // Add some noise
                float noise = (float)(rand() % 100) / 100.0f;
                r = (unsigned char)(r * (0.9f + noise * 0.2f));
                g = (unsigned char)(g * (0.9f + noise * 0.2f));
                b = (unsigned char)(b * (0.9f + noise * 0.2f));
                
                // Set the pixel
                int index = (y * width + x) * 4;
                data[index] = r;
                data[index + 1] = g;
                data[index + 2] = b;
                data[index + 3] = 255; // Alpha
            }
        }
    }
    
    // Generate a concrete texture for buildings
    static void generateConcreteTexture(std::vector<unsigned char>& data, int width, int height) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // Base gray color with slight color variation
                unsigned char val = 180 + (rand() % 40);
                unsigned char r = val - (rand() % 15);
                unsigned char g = val - (rand() % 15);
                unsigned char b = val;
                
                // Add some noise for concrete imperfections
                float noise = (float)(rand() % 100) / 100.0f;
                r = (unsigned char)(r * (0.95f + noise * 0.1f));
                g = (unsigned char)(g * (0.95f + noise * 0.1f));
                b = (unsigned char)(b * (0.95f + noise * 0.1f));
                
                // Set the pixel
                int index = (y * width + x) * 4;
                data[index] = r;
                data[index + 1] = g;
                data[index + 2] = b;
                data[index + 3] = 255; // Alpha
            }
        }
    }
    
    static void generateWindowTexture(std::vector<unsigned char>& data, int width, int height) {
        // Number of windows horizontally and vertically - HALVED for larger windows
        int numWindowsH = 2; // Changed from 4
        int numWindowsV = 4; // Changed from 8
        
        // Window size - doubled
        int windowW = width / numWindowsH;
        int windowH = height / numWindowsV;
        
        // Window border width - proportionally increased
        int borderW = windowW / 10;
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int windowX = x / windowW;
                int windowY = y / windowH;
                
                // Calculate position within the window
                int relX = x % windowW;
                int relY = y % windowH;
                
                // Default color (building wall)
                unsigned char r = 180 + (rand() % 20);
                unsigned char g = 180 + (rand() % 20);
                unsigned char b = 180 + (rand() % 20);
                
                // Check if we're on a window border
                if (relX < borderW || relX >= windowW - borderW || 
                    relY < borderW || relY >= windowH - borderW) {
                    // Window frame (darker)
                    r = 100 + (rand() % 20);
                    g = 100 + (rand() % 20);
                    b = 100 + (rand() % 20);
                } else {
                    // Window glass (blue-ish with random lighting)
                    if (rand() % 10 > 7) { // Some windows are lit
                        r = 220 + (rand() % 35);
                        g = 220 + (rand() % 35);
                        b = 180 + (rand() % 35);
                    } else {
                        r = 70 + (rand() % 20);
                        g = 90 + (rand() % 30);
                        b = 120 + (rand() % 40);
                    }
                }
                
                // Set the pixel
                int index = (y * width + x) * 4;
                data[index] = r;
                data[index + 1] = g;
                data[index + 2] = b;
                data[index + 3] = 255; // Alpha
            }
        }
    }
    
    // Generate a road texture
    static void generateRoadTexture(std::vector<unsigned char>& data, int width, int height) {
        // Calculate the center line position
        int centerLine = height / 2;
        int lineWidth = height / 16;  // Slightly wider line
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // Base asphalt color (dark gray)
                unsigned char val = 80 + (rand() % 20);
                
                // Add some noise for asphalt texture
                float noise = (float)(rand() % 100) / 100.0f;
                val = (unsigned char)(val * (0.95f + noise * 0.1f));
                
                // Draw the center line (yellow)
                if (y > centerLine - lineWidth/2 && y < centerLine + lineWidth/2) {
                    // Add dashed yellow line in the middle
                    int segment = x / (width / 20); // 20 segments
                    if (segment % 2 == 0) { // Draw line only on even segments
                        // Yellow color
                        int index = (y * width + x) * 4;
                        data[index] = 220 + (rand() % 35);     // R - brighter yellow
                        data[index + 1] = 220 + (rand() % 35); // G - brighter yellow
                        data[index + 2] = 0;                   // B - no blue for pure yellow
                        data[index + 3] = 255;                 // Alpha
                        continue;
                    }
                }
                
                // Optional: Add road edge marking
                if ((y < height/10) || (y > height*9/10)) {
                    // White edge marking
                    val = 200 + (rand() % 55);
                }
                
                // Set the pixel to gray asphalt
                int index = (y * width + x) * 4;
                data[index] = val;
                data[index + 1] = val;
                data[index + 2] = val;
                data[index + 3] = 255; // Alpha
            }
        }
    }

    
    // Generate a tree texture (for tree billboards)
    static void generateTreeTexture(std::vector<unsigned char>& data, int width, int height) {
        // Define the tree shape
        int trunkWidth = width / 6;
        int trunkHeight = height / 2;
        int trunkBottom = height - 1;
        int trunkTop = trunkBottom - trunkHeight;
        
        // Center of the trunk
        int trunkCenter = width / 2;
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // Default: transparent
                unsigned char r = 0, g = 0, b = 0, a = 0;
                
                // Tree trunk (brown)
                if (y >= trunkTop && y <= trunkBottom && 
                    x >= trunkCenter - trunkWidth/2 && x <= trunkCenter + trunkWidth/2) {
                    
                    r = 100 + (rand() % 30);
                    g = 70 + (rand() % 20);
                    b = 30 + (rand() % 20);
                    a = 255;
                }
                
                // Tree crown (green circle)
                float distFromCenter = sqrt(
                    pow((x - trunkCenter), 2) + 
                    pow((y - trunkTop/2), 2)
                );
                float crownRadius = trunkTop / 1.5;
                
                if (distFromCenter < crownRadius) {
                    // Green foliage with variation
                    r = 30 + (rand() % 40);
                    g = 100 + (rand() % 80);
                    b = 30 + (rand() % 20);
                    
                    // Edge fadeout for more natural look
                    if (distFromCenter > crownRadius * 0.8) {
                        float fadeout = 1.0f - ((distFromCenter - crownRadius * 0.8) / (crownRadius * 0.2));
                        a = (unsigned char)(255 * fadeout);
                    } else {
                        a = 255;
                    }
                }
                
                // Set the pixel
                int index = (y * width + x) * 4;
                data[index] = r;
                data[index + 1] = g;
                data[index + 2] = b;
                data[index + 3] = a;
            }
        }
    }
    
    // Generate a car texture (for car billboards)
    static void generateCarTexture(std::vector<unsigned char>& data, int width, int height) {
        // Define car shape parameters
        int carWidth = width * 2 / 3;
        int carHeight = height / 3;
        int carBottom = height - 10;
        int carTop = carBottom - carHeight;
        
        // Center of the car
        int carCenter = width / 2;
        
        // Car colors - randomly pick one
        unsigned char carColors[5][3] = {
            {200, 30, 30},   // Red
            {30, 30, 200},   // Blue
            {30, 30, 30},    // Black
            {200, 200, 200}, // White
            {200, 160, 30}   // Yellow
        };
        
        int colorIndex = rand() % 5;
        unsigned char baseR = carColors[colorIndex][0];
        unsigned char baseG = carColors[colorIndex][1];
        unsigned char baseB = carColors[colorIndex][2];
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // Default: transparent
                unsigned char r = 0, g = 0, b = 0, a = 0;
                
                // Car body
                if (y >= carTop && y <= carBottom && 
                    x >= carCenter - carWidth/2 && x <= carCenter + carWidth/2) {
                    
                    // Car color with slight variation
                    float noise = (float)(rand() % 100) / 100.0f * 0.2f;
                    r = (unsigned char)(baseR * (0.9f + noise));
                    g = (unsigned char)(baseG * (0.9f + noise));
                    b = (unsigned char)(baseB * (0.9f + noise));
                    a = 255;
                    
                    // Car windows (top portion)
                    if (y <= carTop + carHeight * 0.5) {
                        r = 100;
                        g = 120;
                        b = 150;
                    }
                    
                    // Wheels
                    int wheelRadius = carHeight / 4;
                    int leftWheelX = carCenter - carWidth/3;
                    int rightWheelX = carCenter + carWidth/3;
                    int wheelY = carBottom - wheelRadius/2;
                    
                    if ((pow(x - leftWheelX, 2) + pow(y - wheelY, 2) < pow(wheelRadius, 2)) ||
                        (pow(x - rightWheelX, 2) + pow(y - wheelY, 2) < pow(wheelRadius, 2))) {
                        r = 30;
                        g = 30;
                        b = 30;
                    }
                }
                
                // Set the pixel
                int index = (y * width + x) * 4;
                data[index] = r;
                data[index + 1] = g;
                data[index + 2] = b;
                data[index + 3] = a;
            }
        }
    }
    
    // Generate a default checker texture
    static void generateDefaultTexture(std::vector<unsigned char>& data, int width, int height) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // Checker pattern
                bool isEven = ((x / 32) + (y / 32)) % 2 == 0;
                
                // Set the pixel
                int index = (y * width + x) * 4;
                if (isEven) {
                    data[index] = 255;     // R
                    data[index + 1] = 0;   // G
                    data[index + 2] = 255; // B
                } else {
                    data[index] = 0;       // R
                    data[index + 1] = 0;   // G
                    data[index + 2] = 0;   // B
                }
                data[index + 3] = 255; // Alpha
            }
        }
    }
};

#endif // _TEXTURE_MANAGER_H_