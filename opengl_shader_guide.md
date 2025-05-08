# OpenGL Shader Guide for MapCraft

This document explains the different shader types used in the MapCraft project and provides a quick reference for GLSL syntax and usage.

## Shader Types Overview

The MapCraft project uses several specialized shaders for different rendering tasks:

| Shader Program | Purpose |
|----------------|---------|
| `shader_program` | Basic rendering without lighting |
| `phong_shader_program` | Realistic lighting with Phong model |
| `tess_shader_program` | Dynamic level of detail with tessellation |
| `water_shader_program` | Realistic water with waves and reflections |
| `shadow_shader_program` | Shadow mapping for realistic shadows |
| `terrain_shader_program` | Terrain rendering with height-based texturing |
| `particle_shader_program` | Particle effects for rain and weather |

## Basic Shader Concepts

### GLSL Version Declaration

All modern shaders begin with a version declaration:

```glsl
#version 330 core  // OpenGL 3.3 core profile
```

### Input Variables

Inputs are declared using the `in` keyword:

```glsl
// Vertex shader inputs (from vertex attributes)
layout (location = 0) in vec3 aPos;       // Position
layout (location = 1) in vec3 aNormal;    // Normal
layout (location = 2) in vec2 aTexCoord;  // Texture coordinates

// Fragment shader inputs (from vertex shader)
in vec3 FragPos;   // World position
in vec3 Normal;    // Normal vector
in vec2 TexCoord;  // Texture coordinates
```

### Output Variables

Outputs are declared using the `out` keyword:

```glsl
// Vertex shader outputs
out vec3 FragPos;   // Position in world space
out vec3 Normal;    // Normal vector
out vec2 TexCoord;  // Texture coordinates

// Fragment shader output
out vec4 FragColor; // Final pixel color (RGBA)
```

### Uniform Variables

Uniforms are values that stay constant for all vertices/fragments in a draw call:

```glsl
// Transformation matrices
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

// Material properties
uniform vec3 objectColor;
uniform float ambient;
uniform float diffuse;
uniform float specular;
uniform float shininess;

// Lighting
uniform vec3 lightPos;
uniform vec3 lightColor;
uniform vec3 viewPos;

// Textures
uniform sampler2D mainTexture;
```

## Basic Shader

The basic shader provides simple rendering without lighting effects:

### Vertex Shader

```glsl
#version 330 core
layout (location = 0) in vec3 aPos;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
void main() {
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
```

### Fragment Shader

```glsl
#version 330 core
out vec4 FragColor;
uniform vec3 color;
void main() {
    FragColor = vec4(color, 1.0);
}
```

## Phong Lighting Shader

The Phong shader implements a lighting model with ambient, diffuse, and specular components:

### Vertex Shader

```glsl
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
```

### Fragment Shader

```glsl
#version 330 core
out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;

// Light properties
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
    // Ambient component
    vec3 ambient_component = ambient * lightColor;
  	
    // Diffuse component
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse_component = diffuse * diff * lightColor;
    
    // Specular component
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);  
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
    vec3 specular_component = specular * spec * lightColor;  
    
    // Combine components
    vec3 result = (ambient_component + diffuse_component + specular_component) * objectColor;
    FragColor = vec4(result, 1.0);
}
```

## Tessellation Shader

The tessellation shader provides dynamic level of detail:

### Vertex Shader

```glsl
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
```

### Tessellation Control Shader

```glsl
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
```

### Tessellation Evaluation Shader

```glsl
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
```

## Water Shader

The water shader creates realistic water effects:

### Vertex Shader

```glsl
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec3 FragPos;
out vec2 TexCoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform float time;
uniform float waveStrength;
uniform float waveFrequency;

void main() {
    TexCoord = aTexCoord;
    
    // Apply wave animation to Y coordinate
    vec3 pos = aPos;
    pos.y += sin(aTexCoord.x * waveFrequency + time) * waveStrength;
    pos.y += cos(aTexCoord.y * waveFrequency + time) * waveStrength;
    
    FragPos = vec3(model * vec4(pos, 1.0));
    gl_Position = projection * view * model * vec4(pos, 1.0);
}
```

### Fragment Shader

```glsl
#version 330 core
out vec4 FragColor;

in vec3 FragPos;
in vec2 TexCoord;

uniform vec3 waterColor;
uniform vec3 lightPos;
uniform vec3 viewPos;
uniform float reflectivity;
uniform float refractionStrength;
uniform float specularPower;

void main() {
    // Lighting calculations
    vec3 norm = vec3(0.0, 1.0, 0.0); // Water surface normal (up)
    vec3 lightDir = normalize(lightPos - FragPos);
    
    // Diffuse
    float diff = max(dot(norm, lightDir), 0.0);
    
    // Specular
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), specularPower);
    
    // Water depth effect based on position
    float depth = clamp(1.0 - (viewPos.y - FragPos.y) / 20.0, 0.0, 1.0);
    
    // Combine effects
    vec3 color = waterColor * (0.3 + diff * 0.7);
    color += vec3(1.0) * spec * 0.5; // Add specular highlight
    
    // Transparency based on viewing angle (more transparent at steep angles)
    float alpha = 0.8 - 0.5 * max(dot(viewDir, norm), 0.0);
    alpha = clamp(alpha + depth * 0.5, 0.4, 0.95);
    
    FragColor = vec4(color, alpha);
}
```

## Shadow Mapping Shader

The shadow mapping process uses two shader programs:

### Depth Map Vertex Shader (First Pass)

```glsl
#version 330 core
layout (location = 0) in vec3 aPos;

uniform mat4 lightSpaceMatrix;
uniform mat4 model;

void main() {
    gl_Position = lightSpaceMatrix * model * vec4(aPos, 1.0);
}
```

### Main Rendering with Shadows (Second Pass)

```glsl
#version 330 core
out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;
in vec4 FragPosLightSpace;

uniform sampler2D shadowMap;
uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec3 lightColor;
uniform vec3 objectColor;

// Calculate shadow value (0.0 = fully lit, 1.0 = fully in shadow)
float ShadowCalculation(vec4 fragPosLightSpace) {
    // Perspective divide
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    
    // Transform to [0,1] range
    projCoords = projCoords * 0.5 + 0.5;
    
    // Get closest depth from light's perspective
    float closestDepth = texture(shadowMap, projCoords.xy).r; 
    
    // Get current depth
    float currentDepth = projCoords.z;
    
    // Calculate bias to prevent shadow acne
    vec3 normal = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float bias = max(0.05 * (1.0 - dot(normal, lightDir)), 0.005);
    
    // Check if fragment is in shadow
    float shadow = currentDepth - bias > closestDepth ? 1.0 : 0.0;
    
    return shadow;
}

void main() {
    // Phong lighting calculations
    vec3 color = objectColor;
    vec3 normal = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, normal);
    
    // Ambient
    vec3 ambient = 0.3 * color;
    
    // Diffuse
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;
    
    // Specular
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = spec * lightColor;
    
    // Calculate shadow
    float shadow = ShadowCalculation(FragPosLightSpace);
    
    // Apply shadow to diffuse and specular components
    vec3 lighting = ambient + (1.0 - shadow) * (diffuse + specular) * color;
    
    FragColor = vec4(lighting, 1.0);
}
```

## GLSL Built-in Functions Reference

### Vector Operations

```glsl
// Normalization
vec3 normalized = normalize(vector);

// Dot product
float dotProduct = dot(vec1, vec2);

// Cross product
vec3 crossProduct = cross(vec1, vec2);

// Length
float length = length(vector);

// Distance
float dist = distance(point1, point2);

// Reflection
vec3 reflected = reflect(incident, normal);
```

### Math Functions

```glsl
// Interpolation
float blended = mix(value1, value2, factor);

// Clamping
float clamped = clamp(value, min, max);

// Trigonometry
float sineValue = sin(angle);
float cosineValue = cos(angle);
float tangent = tan(angle);

// Exponential
float power = pow(base, exponent);
```

### Texture Functions

```glsl
// Basic texture lookup
vec4 color = texture(sampler, texCoords);

// Lookup with LOD
vec4 color = textureLod(sampler, texCoords, lod);

// Projective texture lookup
vec4 color = textureProj(sampler, texCoords);
```

## Debugging Shaders

1. Check for compile errors using `glGetShaderInfoLog`

2. Add debug output to fragment shader:
   ```glsl
   // Output a solid color for debugging
   FragColor = vec4(1.0, 0.0, 0.0, 1.0); // Solid red

   // Visualize normal vectors
   FragColor = vec4(normalize(Normal) * 0.5 + 0.5, 1.0);

   // Visualize texture coordinates
   FragColor = vec4(TexCoord, 0.0, 1.0);
   ```

3. Use `gl_FragCoord` for position-based debugging:
   ```glsl
   // Visualize screen position
   FragColor = vec4(gl_FragCoord.x / 1024.0, gl_FragCoord.y / 768.0, 0.0, 1.0);
   ```

## Performance Considerations

1. **Avoid branching**: Conditional statements in shaders can be expensive
   ```glsl
   // Instead of if/else, use mix:
   vec3 result = mix(colorA, colorB, condition ? 1.0 : 0.0);
   ```

2. **Pre-compute on CPU**: When possible, calculate values on CPU instead of GPU
   ```glsl
   // Instead of this in shader:
   mat3 normalMatrix = mat3(transpose(inverse(model)));
   
   // Pre-compute on CPU and pass as uniform
   ```

3. **Optimize math operations**: Use built-in functions and simplify expressions
   ```glsl
   // Use built-in functions when available
   vec3 normalized = normalize(vector); // Better than vector / length(vector)
   