#include "3d_shaders.h"

// Original shaders
const char* basic_vertex_shader = R"glsl(
#version 330 core
layout (location = 0) in vec3 aPos;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
void main() {
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
)glsl";

const char* basic_fragment_shader = R"glsl(
#version 330 core
out vec4 FragColor;
uniform vec3 color;
void main() {
    FragColor = vec4(color, 1.0);
}
)glsl";

// Phong lighting shaders
const char* phong_vertex_shader = R"glsl(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

out vec3 FragPos;
out vec3 Normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;  
    
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
)glsl";

const char* phong_fragment_shader = R"glsl(
#version 330 core
out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;

uniform vec3 viewPos;
uniform vec3 lightPos;
uniform vec3 lightColor;
uniform vec3 objectColor;

// Material properties
uniform float ambient;
uniform float diffuse;
uniform float specular;
uniform float shininess;

void main() {
    // Ambient
    vec3 ambient_component = ambient * lightColor;
  	
    // Diffuse 
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse_component = diffuse * diff * lightColor;
    
    // Specular
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);  
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
    vec3 specular_component = specular * spec * lightColor;  
    
    // Combine components
    vec3 result = (ambient_component + diffuse_component + specular_component) * objectColor;
    FragColor = vec4(result, 1.0);
}
)glsl";

// Tessellation shaders
const char* tess_vertex_shader = R"glsl(
#version 410 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

out VS_OUT {
    vec3 position;
    vec3 normal;
} vs_out;

void main() {
    vs_out.position = aPos;
    vs_out.normal = aNormal;
}
)glsl";

const char* tess_control_shader = R"glsl(
#version 410 core
layout (vertices = 3) out;

in VS_OUT {
    vec3 position;
    vec3 normal;
} tcs_in[];

out TCS_OUT {
    vec3 position;
    vec3 normal;
} tcs_out[];

uniform float tessellation_level;

void main() {
    // Pass attributes through
    tcs_out[gl_InvocationID].position = tcs_in[gl_InvocationID].position;
    tcs_out[gl_InvocationID].normal = tcs_in[gl_InvocationID].normal;
    
    // Set tessellation levels
    if (gl_InvocationID == 0) {
        gl_TessLevelOuter[0] = tessellation_level;
        gl_TessLevelOuter[1] = tessellation_level;
        gl_TessLevelOuter[2] = tessellation_level;
        
        gl_TessLevelInner[0] = tessellation_level;
    }
}
)glsl";

const char* tess_evaluation_shader = R"glsl(
#version 410 core
layout (triangles, equal_spacing, ccw) in;

in TCS_OUT {
    vec3 position;
    vec3 normal;
} tes_in[];

out TES_OUT {
    vec3 position;
    vec3 normal;
    vec3 world_pos;
} tes_out;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    // Barycentric coordinates
    vec3 p0 = gl_TessCoord.x * tes_in[0].position;
    vec3 p1 = gl_TessCoord.y * tes_in[1].position;
    vec3 p2 = gl_TessCoord.z * tes_in[2].position;
    
    // Interpolate position
    tes_out.position = p0 + p1 + p2;
    
    // Interpolate normal
    vec3 n0 = gl_TessCoord.x * tes_in[0].normal;
    vec3 n1 = gl_TessCoord.y * tes_in[1].normal;
    vec3 n2 = gl_TessCoord.z * tes_in[2].normal;
    tes_out.normal = normalize(n0 + n1 + n2);
    
    // Calculate world position
    tes_out.world_pos = vec3(model * vec4(tes_out.position, 1.0));
    
    // Calculate clip space position
    gl_Position = projection * view * model * vec4(tes_out.position, 1.0);
}
)glsl";

const char* tess_fragment_shader = R"glsl(
#version 410 core
out vec4 FragColor;

in TES_OUT {
    vec3 position;
    vec3 normal;
    vec3 world_pos;
} fs_in;

uniform vec3 viewPos;
uniform vec3 lightPos;
uniform vec3 lightColor;
uniform vec3 objectColor;

// Material properties
uniform float ambient;
uniform float diffuse;
uniform float specular;
uniform float shininess;

void main() {
    // Ambient
    vec3 ambient_component = ambient * lightColor;
  	
    // Diffuse 
    vec3 norm = normalize(fs_in.normal);
    vec3 lightDir = normalize(lightPos - fs_in.world_pos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse_component = diffuse * diff * lightColor;
    
    // Specular
    vec3 viewDir = normalize(viewPos - fs_in.world_pos);
    vec3 reflectDir = reflect(-lightDir, norm);  
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
    vec3 specular_component = specular * spec * lightColor;  
    
    // Combine components
    vec3 result = (ambient_component + diffuse_component + specular_component) * objectColor;
    FragColor = vec4(result, 1.0);
}
)glsl";

// Water shaders
const char* water_vertex_shader = R"glsl(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 TexCoord;
out vec3 FragPos;
out vec3 Normal;
out vec4 ClipSpace;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform float time;
uniform vec3 cameraPosition;

// Water wave parameters
uniform float waveStrength;
uniform float waveSpeed;
uniform float waveFrequency;

// Calculate Gerstner wave
vec3 calculateGerstnerWave(vec3 position, float time) {
    vec3 result = position;
    
    // Direction of the wave (varies based on sample position for more natural look)
    vec2 direction1 = normalize(vec2(sin(position.x * 0.1), cos(position.z * 0.1)));
    vec2 direction2 = normalize(vec2(cos(position.x * 0.2), sin(position.z * 0.2)));
    
    // Wave parameters
    float steepness = 0.3;
    float wavelength1 = 20.0;
    float wavelength2 = 15.0;
    float k1 = 2.0 * 3.14159 / wavelength1;
    float k2 = 2.0 * 3.14159 / wavelength2;
    float speed1 = 1.0;
    float speed2 = 1.3;
    float a1 = waveStrength;
    float a2 = waveStrength * 0.5;
    
    // First wave
    float f1 = k1 * (dot(direction1, vec2(position.x, position.z)) - speed1 * time * waveSpeed);
    result.x += steepness * a1 * direction1.x * cos(f1);
    result.z += steepness * a1 * direction1.y * cos(f1);
    result.y += a1 * sin(f1);
    
    // Second wave
    float f2 = k2 * (dot(direction2, vec2(position.x, position.z)) - speed2 * time * waveSpeed);
    result.x += steepness * a2 * direction2.x * cos(f2);
    result.z += steepness * a2 * direction2.y * cos(f2);
    result.y += a2 * sin(f2);
    
    return result;
}

void main() {
    // Apply Gerstner waves
    vec3 wavePos = calculateGerstnerWave(aPos, time);
    
    // Output all necessary variables
    FragPos = vec3(model * vec4(wavePos, 1.0));
    TexCoord = aTexCoord;
    ClipSpace = projection * view * model * vec4(wavePos, 1.0);
    
    // Calculate normal based on wave
    float epsilon = 0.01;
    vec3 tangentX = calculateGerstnerWave(aPos + vec3(epsilon, 0.0, 0.0), time) - calculateGerstnerWave(aPos - vec3(epsilon, 0.0, 0.0), time);
    vec3 tangentZ = calculateGerstnerWave(aPos + vec3(0.0, 0.0, epsilon), time) - calculateGerstnerWave(aPos - vec3(0.0, 0.0, epsilon), time);
    Normal = normalize(cross(tangentX, tangentZ));
    
    gl_Position = ClipSpace;
}
)glsl";

const char* water_fragment_shader = R"glsl(
#version 330 core
out vec4 FragColor;

in vec2 TexCoord;
in vec3 FragPos;
in vec3 Normal;
in vec4 ClipSpace;

uniform vec3 waterColor;
uniform vec3 lightPos;
uniform vec3 viewPos;
uniform float time;

// Water properties
uniform float reflectivity;
uniform float refractionStrength;
uniform float specularPower;

void main() {
    // Base water color
    vec3 baseColor = waterColor;
    
    // Normalize vectors
    vec3 normal = normalize(Normal);
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 lightDir = normalize(lightPos - FragPos);
    
    // Calculate Fresnel factor (more reflective at grazing angles)
    float fresnel = pow(1.0 - max(dot(normal, viewDir), 0.0), 5.0) * reflectivity;
    
    // Add some ripple effect to the water by perturbing the normal
    float ripple = sin(TexCoord.x * 50.0 + time) * sin(TexCoord.y * 50.0 + time) * 0.05;
    normal = normalize(normal + vec3(ripple, 0.0, ripple));
    
    // Diffuse lighting
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * vec3(1.0);
    
    // Specular lighting (water highlights)
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), specularPower);
    vec3 specular = spec * vec3(1.0, 1.0, 1.0);
    
    // Combine components
    vec3 result = baseColor * 0.3 + baseColor * diffuse * 0.5 + specular * 0.8;
    
    // Adjust color based on fresnel factor (more blue in deeper areas)
    result = mix(result, baseColor * 0.5, 1.0 - fresnel);
    
    // Add transparency effect
    float alpha = 0.8 + fresnel * 0.15;
    
    FragColor = vec4(result, alpha);
}
)glsl";

// Shadow mapping shaders
const char* shadow_mapping_vertex_shader = R"glsl(
#version 330 core
layout (location = 0) in vec3 aPos;

uniform mat4 lightSpaceMatrix;
uniform mat4 model;

void main() {
    gl_Position = lightSpaceMatrix * model * vec4(aPos, 1.0);
}
)glsl";

const char* shadow_mapping_fragment_shader = R"glsl(
#version 330 core
void main() {
    // This is intentionally empty - shadow maps only need depth
}
)glsl";

// Shadow receiver vertex shader
const char* shadow_receiver_vertex_shader = R"glsl(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

out VS_OUT {
    vec3 FragPos;
    vec3 Normal;
    vec4 FragPosLightSpace;
} vs_out;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;
uniform mat4 lightSpaceMatrix;

void main() {
    vs_out.FragPos = vec3(model * vec4(aPos, 1.0));
    vs_out.Normal = transpose(inverse(mat3(model))) * aNormal;
    vs_out.FragPosLightSpace = lightSpaceMatrix * vec4(vs_out.FragPos, 1.0);
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
)glsl";

const char* shadow_receiver_fragment_shader = R"glsl(
#version 330 core
out vec4 FragColor;

in VS_OUT {
    vec3 FragPos;
    vec3 Normal;
    vec4 FragPosLightSpace;
} fs_in;

uniform sampler2D shadowMap;
uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec3 objectColor;
uniform float ambient;
uniform float diffuse;
uniform float specular;
uniform float shininess;

float ShadowCalculation(vec4 fragPosLightSpace) {
    // Perform perspective divide
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    
    // Transform to [0,1] range
    projCoords = projCoords * 0.5 + 0.5;
    
    // Get closest depth from light's perspective
    float closestDepth = texture(shadowMap, projCoords.xy).r; 
    
    // Get current depth in light space
    float currentDepth = projCoords.z;
    
    // Add bias to prevent shadow acne
    vec3 normal = normalize(fs_in.Normal);
    vec3 lightDir = normalize(lightPos - fs_in.FragPos);
    float bias = max(0.05 * (1.0 - dot(normal, lightDir)), 0.005);
    
    // PCF (percentage-closer filtering) for smoother shadows
    float shadow = 0.0;
    vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
    for(int x = -1; x <= 1; ++x) {
        for(int y = -1; y <= 1; ++y) {
            float pcfDepth = texture(shadowMap, projCoords.xy + vec2(x, y) * texelSize).r; 
            shadow += currentDepth - bias > pcfDepth ? 1.0 : 0.0;        
        }    
    }
    shadow /= 9.0;
    
    // Ensure we don't cast shadows beyond the far plane
    if(projCoords.z > 1.0)
        shadow = 0.0;
        
    return shadow;
}

void main() {
    // Ambient lighting
    vec3 ambient_light = vec3(ambient);
    
    // Diffuse lighting
    vec3 normal = normalize(fs_in.Normal);
    vec3 lightDir = normalize(lightPos - fs_in.FragPos);
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse_light = vec3(diffuse) * diff;
    
    // Specular lighting
    vec3 viewDir = normalize(viewPos - fs_in.FragPos);
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
    vec3 specular_light = vec3(specular) * spec;
    
    // Calculate shadow
    float shadow = ShadowCalculation(fs_in.FragPosLightSpace);
    
    // Final color (ambient is not affected by shadow)
    vec3 result = (ambient_light + (1.0 - shadow) * (diffuse_light + specular_light)) * objectColor;
    
    FragColor = vec4(result, 1.0);
}
)glsl";

// Perlin noise implementation for terrain
const char* perlin_noise_function = R"glsl(
// Perlin Noise functions in GLSL
float rand(vec2 co) {
    return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

vec2 rand2(vec2 p) {
    return fract(vec2(sin(p.x * 591.32 + p.y * 154.077), cos(p.x * 391.32 + p.y * 49.077)));
}

float noise(vec2 p) {
    vec2 ip = floor(p);
    vec2 u = fract(p);
    
    // Improve the noise pattern with smoother interpolation
    u = u * u * (3.0 - 2.0 * u);
    
    float res = mix(
        mix(rand(ip), rand(ip + vec2(1.0, 0.0)), u.x),
        mix(rand(ip + vec2(0.0, 1.0)), rand(ip + vec2(1.0, 1.0)), u.x), u.y);
    return res * res;
}

float fbm(vec2 p) {
    float f = 0.0;
    float w = 0.5;
    for (int i = 0; i < 5; i++) {
        f += w * noise(p);
        p *= 2.0;
        w *= 0.5;
    }
    return f;
}
)glsl";

std::string terrain_vertex_shader_str = std::string(R"glsl(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 TexCoord;
out vec3 FragPos;
out vec3 Normal;
out vec4 FragPosLightSpace;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat4 lightSpaceMatrix;

// Noise parameters
uniform float noiseScale;
uniform float noiseHeight;
uniform vec2 noiseOffset;

// Perlin noise functions
)glsl") + std::string(perlin_noise_function) + std::string(R"glsl(

void main() {
    TexCoord = aTexCoord;
    
    // Apply Perlin noise to y coordinate
    vec2 noisePos = (vec2(aPos.x, aPos.z) + noiseOffset) * noiseScale;
    float noise_val = fbm(noisePos);
    
    // Create the modified position with noise-based height
    vec3 position = aPos;
    position.y += noise_val * noiseHeight;
    
    // Calculate normal based on height field
    float eps = 0.01;
    float height1 = fbm((vec2(aPos.x + eps, aPos.z) + noiseOffset) * noiseScale) * noiseHeight;
    float height2 = fbm((vec2(aPos.x - eps, aPos.z) + noiseOffset) * noiseScale) * noiseHeight;
    float height3 = fbm((vec2(aPos.x, aPos.z + eps) + noiseOffset) * noiseScale) * noiseHeight;
    float height4 = fbm((vec2(aPos.x, aPos.z - eps) + noiseOffset) * noiseScale) * noiseHeight;
    
    vec3 tangent1 = normalize(vec3(2.0 * eps, height1 - height2, 0.0));
    vec3 tangent2 = normalize(vec3(0.0, height3 - height4, 2.0 * eps));
    Normal = normalize(cross(tangent1, tangent2));
    
    // Set final position and outputs
    FragPos = vec3(model * vec4(position, 1.0));
    FragPosLightSpace = lightSpaceMatrix * vec4(FragPos, 1.0);
    gl_Position = projection * view * model * vec4(position, 1.0);
}
)glsl");

const char* terrain_vertex_shader = terrain_vertex_shader_str.c_str();


// Sky shader for sunset effect
const char* sky_vertex_shader = R"glsl(
#version 330 core
layout (location = 0) in vec3 aPos;

void main() {
    gl_Position = vec4(aPos, 1.0);
}
)glsl";

const char* sky_fragment_shader = R"glsl(
#version 330 core
out vec4 FragColor;

uniform vec3 bottomColor;
uniform vec3 topColor;
uniform float screenHeight;

void main() {
    float t = gl_FragCoord.y / screenHeight;
    vec3 skyColor = mix(bottomColor, topColor, t);
    FragColor = vec4(skyColor, 1.0);
}
)glsl";


const char* terrain_fragment_shader = R"glsl(
#version 330 core
out vec4 FragColor;

in vec2 TexCoord;
in vec3 FragPos;
in vec3 Normal;
in vec4 FragPosLightSpace;

uniform sampler2D shadowMap;
uniform sampler2D grassTexture;
uniform sampler2D rockTexture;
uniform sampler2D soilTexture;

uniform vec3 lightPos;
uniform vec3 viewPos;
uniform float time;

// Material properties
uniform float ambient;
uniform float diffuse;
uniform float specular;
uniform float shininess;

// Terrain parameters
uniform float grassThreshold;    // Height below which is mostly grass
uniform float rockThreshold;     // Height above which is mostly rock
uniform float snowThreshold;     // Height above which is mostly snow
uniform float slopeThreshold;    // Slope angle above which is more rocky

float ShadowCalculation(vec4 fragPosLightSpace) {
    // Perspective divide
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    
    // Transform to [0,1] range
    projCoords = projCoords * 0.5 + 0.5;
    
    // Get depth from shadow map
    float closestDepth = texture(shadowMap, projCoords.xy).r; 
    float currentDepth = projCoords.z;
    
    // Calculate bias based on slope
    vec3 normal = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float bias = max(0.05 * (1.0 - dot(normal, lightDir)), 0.005);
    
    // PCF for softer shadows
    float shadow = 0.0;
    vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
    for(int x = -2; x <= 2; ++x) {
        for(int y = -2; y <= 2; ++y) {
            float pcfDepth = texture(shadowMap, projCoords.xy + vec2(x, y) * texelSize).r; 
            shadow += currentDepth - bias > pcfDepth ? 1.0 : 0.0;        
        }    
    }
    shadow /= 25.0;
    
    if(projCoords.z > 1.0)
        shadow = 0.0;
        
    return shadow;
}

// Procedural noise for terrain detail
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f*f*(3.0-2.0*f); // Smooth interpolation
    
    return mix(
        mix(hash(i), hash(i + vec2(1.0, 0.0)), f.x),
        mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), f.x),
        f.y);
}

void main() {
    // Calculate terrain blending based on height and slope
    float height = FragPos.y;
    float slope = 1.0 - dot(normalize(Normal), vec3(0.0, 1.0, 0.0));
    
    // Add some small-scale detail with procedural noise
    float detail = noise(TexCoord * 100.0) * 0.05; // Small detail variation
    
    // Sample textures with detail
    vec2 detailCoord = TexCoord * 30.0; // Scale for tiling
    vec3 grassColorSample = texture(grassTexture, detailCoord).rgb;
    vec3 rockColorSample = texture(rockTexture, detailCoord).rgb;
    vec3 soilColorSample = texture(soilTexture, detailCoord).rgb;
    
    // Blend based on height and slope
    float grassWeight = 1.0 - smoothstep(grassThreshold, rockThreshold, height) 
                      - slope * slopeThreshold;
    float rockWeight = smoothstep(grassThreshold, rockThreshold, height) 
                    - smoothstep(rockThreshold, snowThreshold, height) 
                    + slope * slopeThreshold;
    float soilWeight = smoothstep(rockThreshold, snowThreshold, height) 
                    - slope * slopeThreshold * 2.0;
    
    // Clamp weights
    grassWeight = clamp(grassWeight, 0.0, 1.0);
    rockWeight = clamp(rockWeight, 0.0, 1.0);
    soilWeight = clamp(soilWeight, 0.0, 1.0);
    
    // Normalize weights
    float totalWeight = grassWeight + rockWeight + soilWeight;
    grassWeight /= totalWeight;
    rockWeight /= totalWeight;
    soilWeight /= totalWeight;
    
    // Calculate final color with texture blending
    vec3 terrainColor = grassColorSample * grassWeight + 
                     rockColorSample * rockWeight + 
                     soilColorSample * soilWeight;
    
    // Add small-scale detail variation
    terrainColor += detail;
    
    // Lighting (Phong model)
    vec3 normal = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, normal);
    
    // Ambient
    vec3 ambient_component = ambient * terrainColor;
    
    // Diffuse
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse_component = diffuse * diff * terrainColor;
    
    // Specular
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
    vec3 specular_component = specular * spec * vec3(1.0);
    
    // Shadow calculation
    float shadow = ShadowCalculation(FragPosLightSpace);
    
    // Final color (ambient light isn't affected by shadow)
    vec3 result = ambient_component + (1.0 - shadow) * (diffuse_component + specular_component);
    
    FragColor = vec4(result, 1.0);
}
)glsl";

// Particle system shaders (for rain)
const char* particle_vertex_shader = R"glsl(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aVelocity;
layout (location = 2) in float aAge;
layout (location = 3) in float aLife;

out float Age;
out float Life;
out vec3 Velocity;
out float Alpha;
out float Size;

uniform mat4 view;
uniform mat4 projection;
uniform float particleSize;

void main() {
    Age = aAge;
    Life = aLife;
    Velocity = aVelocity;
    
    // Calculate alpha based on particle age
    Alpha = 1.0 - (Age / Life);
    
    // Calculate size based on particle age
    Size = particleSize * (0.5 + 0.5 * Alpha);
    
    // Position the particle
    gl_Position = projection * view * vec4(aPos, 1.0);
    
    // Set point size
    gl_PointSize = Size;
}
)glsl";

const char* particle_fragment_shader = R"glsl(
#version 330 core
out vec4 FragColor;

in float Age;
in float Life;
in vec3 Velocity;
in float Alpha;
in float Size;

void main() {
    // Calculate a circular particle
    vec2 coords = gl_PointCoord * 2.0 - 1.0;
    float radius = length(coords);
    
    // Discard pixels outside the circle
    if(radius > 1.0)
        discard;
    
    // Raindrop color with alpha
    vec3 rainColor = vec3(0.7, 0.7, 0.9);  // Slightly blue tint
    
    // Make the particles look like streaks based on velocity
    float streak = pow(1.0 - radius, 3.0);
    
    // Final color with alpha
    FragColor = vec4(rainColor, Alpha * streak);
}
)glsl";



// Add this new shader to support texturing with Phong lighting
const char* phong_textured_fragment_shader = R"glsl(
#version 330 core
out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;

uniform vec3 viewPos;
uniform vec3 lightPos;
uniform vec3 lightColor;
uniform vec3 objectColor;

// Material properties
uniform float ambient;
uniform float diffuse;
uniform float specular;
uniform float shininess;

// Textures
uniform sampler2D mainTexture;
uniform sampler2D windowTexture;
uniform int useTexture;
uniform int useWindows;
uniform float windowDensity;

void main() {
    // Get base color
    vec3 baseColor;
    
    if (useTexture == 1) {
        // Use texture for the base color
        baseColor = texture(mainTexture, TexCoord).rgb;
        
        // Apply object color tint
        baseColor *= objectColor;
        
        // Apply windows if enabled (for buildings)
        if (useWindows == 1) {
            // Check if this is a window location based on texture coordinates
            float windowSize = 0.1; // Size of windows
            float windowSpacing = 0.15; // Spacing between windows
            
            float xMod = mod(TexCoord.x, windowSpacing);
            float yMod = mod(TexCoord.y, windowSpacing);
            
            bool isWindow = xMod < windowSize && yMod < windowSize;
            
            // Apply window density factor
            if (isWindow && mod(float(int(TexCoord.x / windowSpacing) + int(TexCoord.y / windowSpacing)), 4.0) < windowDensity * 4.0) {
                // Sample window texture
                vec3 windowColor = texture(windowTexture, vec2(xMod / windowSize, yMod / windowSize)).rgb;
                
                // Mix window with base color
                baseColor = mix(baseColor, windowColor, 0.8);
                
                // Add some emissive light for lit windows (random pattern)
                if (mod(float(int(TexCoord.x / windowSpacing) * 13 + int(TexCoord.y / windowSpacing) * 7), 4.0) < 1.0) {
                    baseColor *= 1.5; // Brighter for lit windows
                }
            }
        }
    } else {
        // Use solid color
        baseColor = objectColor;
    }
    
    // Ambient
    vec3 ambient_component = ambient * lightColor;
  	
    // Diffuse 
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse_component = diffuse * diff * lightColor;
    
    // Specular
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);  
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
    vec3 specular_component = specular * spec * lightColor;  
    
    // Combine components
    vec3 result = (ambient_component + diffuse_component + specular_component) * baseColor;
    FragColor = vec4(result, 1.0);
}
)glsl";

// Updated vertex shader to pass texture coordinates
const char* phong_textured_vertex_shader = R"glsl(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;

out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;
    TexCoord = aTexCoord;
    
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
)glsl";




// } // namespace Shaders


// Setup OpenGL shaders
// Update the setup_shaders() method to include Phong shaders
void Renderer3D::setup_shaders() {
    // Original shader setup
    GLuint vertex = compile_shader(GL_VERTEX_SHADER, basic_vertex_shader);
    GLuint fragment = compile_shader(GL_FRAGMENT_SHADER, basic_fragment_shader);
    
    shader_program = glCreateProgram();
    glAttachShader(shader_program, vertex);
    glAttachShader(shader_program, fragment);
    glLinkProgram(shader_program);
    
    // Check for linking errors
    GLint success;
    glGetProgramiv(shader_program, GL_LINK_STATUS, &success);
    if (!success) {
        char info_log[512];
        glGetProgramInfoLog(shader_program, 512, nullptr, info_log);
        std::cerr << "Shader program linking failed: " << info_log << std::endl;
        throw std::runtime_error("Shader program linking failed");
    }
    
    // Delete shaders as they're linked into the program now
    glDeleteShader(vertex);
    glDeleteShader(fragment);
    
    // Create vertex array object and vertex buffer object
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    
    // Configure vertex attributes
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    
    // Setup advanced shaders
    setup_phong_shaders();
    setup_tessellation_shaders();
    setup_water_shaders();
    setup_shadow_mapping();
    setup_terrain_shaders();
    setup_particle_system();
    
    std::cout << "All shaders initialized successfully" << std::endl;
}

// Compile individual shader
/**
 * Compiles a GLSL shader from source code
 * 
 * This function demonstrates the OpenGL shader compilation process:
 * 1. Create a shader object of the specified type
 * 2. Attach source code to the shader
 * 3. Compile the shader
 * 4. Check for compilation errors
 * 
 * OpenGL shader types include:
 * - GL_VERTEX_SHADER: Processes each vertex position/attribute
 * - GL_FRAGMENT_SHADER: Processes each pixel (fragment) to determine color
 * - GL_GEOMETRY_SHADER: Optional shader that processes primitives
 * - GL_TESS_CONTROL_SHADER: Controls tessellation levels
 * - GL_TESS_EVALUATION_SHADER: Processes tessellated vertices
 * 
 * @param type GLenum specifying shader type (GL_VERTEX_SHADER, GL_FRAGMENT_SHADER, etc.)
 * @param source C-string containing GLSL shader source code
 * @return GLuint handle to the compiled shader object
 * @throws std::runtime_error if compilation fails
 */
GLuint Renderer3D::compile_shader(GLenum type, const char* source) {
    // glCreateShader - creates empty shader object of specified type
    // - takes shader type enum (vertex, fragment, etc.)
    // - returns handle (id) for the shader object
    // - returns 0 if creation fails
    GLuint shader = glCreateShader(type);
    
    // glShaderSource - sets the source code in a shader
    // - shader - handle to target shader object
    // - 1 - number of strings in the source array
    // - &source - array of source code strings
    // - nullptr - array of string lengths (null = null-terminated)
    glShaderSource(shader, 1, &source, nullptr);
    
    // glCompileShader - compiles the source code in a shader object
    // - compiles glsl code into gpu instructions
    // - operates on currently bound shader
    // - compilation errors checked with glGetShaderiv
    glCompileShader(shader);
    
    // check compilation status
    // - glGetShaderiv retrieves shader parameters
    // - GL_COMPILE_STATUS returns compilation success/failure
    // - success will be GL_TRUE if compilation succeeded
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        // get error information if compilation failed
        // - glGetShaderInfoLog retrieves compiler error messages
        // - 512 - maximum length of error message to retrieve
        // - nullptr - actual length not needed
        // - info_log - buffer to store error message
        char info_log[512];
        glGetShaderInfoLog(shader, 512, nullptr, info_log);
        std::cerr << "shader compilation failed: " << info_log << std::endl;
        throw std::runtime_error("shader compilation failed");
    }
    
    return shader;
}

// Create shader program from vertex and fragment sources
/**
 * Creates a complete shader program from vertex and fragment shader sources
 * 
 * This function demonstrates the OpenGL shader program creation process:
 * 1. Compile individual vertex and fragment shaders
 * 2. Create a program object
 * 3. Attach shaders to the program
 * 4. Link the program
 * 5. Check for linking errors
 * 6. Clean up individual shaders
 * 
 * Shader programs in OpenGL:
 * - Combine multiple shader stages into a pipeline
 * - Allow data to flow between shader stages
 * - Define the complete rendering process
 * 
 * @param vertex_source Source code for the vertex shader
 * @param fragment_source Source code for the fragment shader
 * @return GLuint handle to the linked shader program
 * @throws std::runtime_error if compilation or linking fails
 */
GLuint Renderer3D::create_shader_program(const char* vertex_source, const char* fragment_source) {
    // compile individual shaders
    // - vertex shader processes each vertex position/attributes 
    // - fragment shader determines pixel colors
    GLuint vertex = compile_shader(GL_VERTEX_SHADER, vertex_source);
    GLuint fragment = compile_shader(GL_FRAGMENT_SHADER, fragment_source);
    
    // glCreateProgram - creates empty program container
    // - returns handle to new program object
    // - program objects link multiple shaders together
    // - returns 0 if creation fails
    GLuint program = glCreateProgram();
    
    // glAttachShader - connects shaders to program
    // - program - target program handle
    // - vertex/fragment - compiled shader handles
    // - defines which shader stages will be used
    glAttachShader(program, vertex);
    glAttachShader(program, fragment);
    
    // glLinkProgram - links all attached shaders
    // - resolves all interface variables between stages
    // - validates compatibility between shader stages
    // - creates executable that runs on the gpu
    glLinkProgram(program);
    
    // check for linking errors
    // - glGetProgramiv retrieves program parameters
    // - GL_LINK_STATUS returns linking success/failure
    // - success will be GL_TRUE if linking succeeded
    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        // get error information if linking failed
        // - glGetProgramInfoLog retrieves linker error messages
        // - 512 - maximum length of error message to retrieve
        // - nullptr - actual length not needed
        // - info_log - buffer to store error message
        char info_log[512];
        glGetProgramInfoLog(program, 512, nullptr, info_log);
        std::cerr << "program linking failed: " << info_log << std::endl;
        throw std::runtime_error("program linking failed");
    }
    
    // cleanup individual shaders
    // - glDeleteShader frees gpu resources
    // - shaders are no longer needed after linking
    // - doesn't affect the program that has them attached
    glDeleteShader(vertex);
    glDeleteShader(fragment);
    
    return program;
}

