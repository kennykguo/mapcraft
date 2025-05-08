# glsl shader guide for mapcraft

## shader language basics

### shader types
- vertex shader - processes vertices, projects coordinates
- fragment shader - computes final pixel color
- geometry shader - creates/modifies geometry
- compute shader - general-purpose parallel computation

### basic structure
```glsl
#version 330 core  // shader version declaration

// input variables
in vec3 position;
in vec2 texCoord;

// output variables
out vec2 fragTexCoord;

// uniform variables
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    // shader main function
    fragTexCoord = texCoord;
    gl_Position = projection * view * model * vec4(position, 1.0);
}
```

### data types
- scalar - `float`, `int`, `bool`
- vector - `vec2`, `vec3`, `vec4`, `ivec2`, `bvec3`, etc.
- matrix - `mat2`, `mat3`, `mat4`
- sampler - `sampler2D`, `sampler3D`, `samplerCube`

### vector operations
- component access - `vec.x`, `vec.y`, `vec.z`, `vec.w`
- swizzling - `vec.xy`, `vec.yzw`, `vec.xyzw`, etc.
- mathematical operations - `+`, `-`, `*`, `/`, etc.

### built-in functions
- geometric - `length()`, `distance()`, `normalize()`, `reflect()`
- trigonometric - `sin()`, `cos()`, `tan()`, `asin()`
- interpolation - `mix()`, `smoothstep()`
- mathematical - `pow()`, `exp()`, `log()`, `sqrt()`

## pipeline stages

### vertex shader
- processes each vertex
- transforms coordinates from model space to clip space
- passes attributes to next stage
```glsl
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;

out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoords;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;  
    TexCoords = aTexCoords;
    
    gl_Position = projection * view * vec4(FragPos, 1.0);
}
```

### fragment shader
- processes each fragment (potential pixel)
- calculates final pixel color
- can discard fragments
```glsl
#version 330 core
out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoords;

uniform sampler2D diffuseMap;
uniform vec3 lightPos;
uniform vec3 viewPos;

void main() {
    // ambient
    vec3 ambient = 0.1 * texture(diffuseMap, TexCoords).rgb;
    
    // diffuse
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * texture(diffuseMap, TexCoords).rgb;
    
    // specular
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = 0.3 * spec * vec3(1.0, 1.0, 1.0);
    
    FragColor = vec4(ambient + diffuse + specular, 1.0);
}
```

## key concepts

### attributes
- input variables in vertex shader
- per-vertex data from vertex buffers
- specified with `layout (location = X)` qualifier

### uniforms
- global variables accessible in all shaders
- values set from application
- remain constant during draw call
- typical uniforms - transform matrices, light positions, material properties

### samplers
- special uniform variables for accessing textures
- bound to texture units
- example - `sampler2D diffuseMap`, `samplerCube environmentMap`

### varying/interpolated variables
- pass data between shader stages
- automatically interpolated across primitive
- `out` in vertex shader, `in` in fragment shader

## lighting models

### phong lighting
- ambient component - constant background light
- diffuse component - directional reflection based on surface normal
- specular component - reflection based on view angle
```glsl
// fragment shader
void main() {
    // ambient
    vec3 ambient = ambientStrength * lightColor;
  	
    // diffuse 
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;
    
    // specular
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);  
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
    vec3 specular = specularStrength * spec * lightColor;  
    
    vec3 result = (ambient + diffuse + specular) * objectColor;
    FragColor = vec4(result, 1.0);
}
```

### blinn-phong lighting
- improvement on phong, using half-vector for specular
- more physically accurate
```glsl
// fragment shader specular calculation with blinn-phong
vec3 viewDir = normalize(viewPos - FragPos);
vec3 lightDir = normalize(lightPos - FragPos);
vec3 halfwayDir = normalize(lightDir + viewDir);
float spec = pow(max(dot(normal, halfwayDir), 0.0), shininess);
vec3 specular = specularStrength * spec * lightColor;
```

### pbr (physically based rendering)
- simulates real-world light behavior
- energy conservation, microfacet theory
- parameters - albedo, metallic, roughness, ao
- more complex but realistic results

## texture mapping

### texture sampling
```glsl
// basic sampling
vec4 texColor = texture(diffuseMap, TexCoords);

// with offset
vec4 texColor = texture(diffuseMap, TexCoords + offset);

// with projection (shadow mapping)
vec4 projCoords = lightSpaceMatrix * vec4(FragPos, 1.0);
projCoords.xyz /= projCoords.w;
projCoords.xyz = projCoords.xyz * 0.5 + 0.5;
float closestDepth = texture(shadowMap, projCoords.xy).r;
```

### texture coordinates
- generated with various methods
- planar mapping, cylindrical mapping, spherical mapping
- triplanar mapping for complex surfaces

### texture blending
- mix multiple textures based on weights
```glsl
// blend between grass and rock based on height
float weight = smoothstep(0.3, 0.5, worldPos.y);
vec4 grassColor = texture(grassTexture, TexCoords);
vec4 rockColor = texture(rockTexture, TexCoords);
vec4 finalColor = mix(grassColor, rockColor, weight);
```

## special effects

### shadow mapping
- render depth from light's perspective
- compare fragment depth with stored shadow map
```glsl
float ShadowCalculation(vec4 fragPosLightSpace) {
    // perform perspective divide
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    
    // transform to [0,1] range
    projCoords = projCoords * 0.5 + 0.5;
    
    // get closest depth value from light's perspective
    float closestDepth = texture(shadowMap, projCoords.xy).r; 
    
    // get current depth value
    float currentDepth = projCoords.z;
    
    // check if fragment is in shadow
    float shadow = currentDepth > closestDepth ? 1.0 : 0.0;

    return shadow;
}
```

### water simulation
- combination of normal mapping, reflection, refraction
- animated normal maps for waves
- fresnel effect for realistic water appearance
```glsl
// water fragment shader excerpt
// calculate reflection/refraction coordinates
vec4 reflectPos = projection * view * vec4(reflect(viewDir, normal), 0.0);
vec4 refractPos = projection * view * vec4(refract(viewDir, normal, 0.75), 0.0);

// sample reflection and refraction textures
vec4 reflectColor = texture(reflectionTex, reflectPos.xy);
vec4 refractColor = texture(refractionTex, refractPos.xy);

// calculate fresnel factor
float fresnel = max(0.0, min(1.0, pow(1.0 - dot(viewDir, normal), 3.0)));

// blend based on fresnel
vec4 finalColor = mix(refractColor, reflectColor, fresnel);
```

### terrain rendering
- heightmap-based displacement
- multi-texture blending based on height/slope
- tessellation for dynamic level of detail
```glsl
// terrain vertex shader with displacement
vec4 worldPos = model * vec4(position, 1.0);
float height = texture(heightMap, texCoord).r;
worldPos.y = height * heightScale;
gl_Position = projection * view * worldPos;
```

## procedural techniques

### noise functions
- perlin noise for natural patterns
- simplex noise for more efficient calculation
- combining noise at different frequencies (octaves)
```glsl
// using noise for terrain generation
float n1 = noise(position.xz * 0.1);
float n2 = noise(position.xz * 0.2) * 0.5;
float n3 = noise(position.xz * 0.4) * 0.25;
float n4 = noise(position.xz * 0.8) * 0.125;
float height = n1 + n2 + n3 + n4;
```

### procedural texturing
- generating texture patterns on-the-fly
- cellular noise for organic patterns
- voronoi patterns for cracks, cells
```glsl
// procedural brick pattern
vec2 brick = floor(fragCoord / vec2(brickWidth, brickHeight));
float offsetX = mod(brick.y, 2.0) * 0.5;
vec2 pos = mod(fragCoord / vec2(brickWidth, brickHeight) + vec2(offsetX, 0.0), 1.0);
float edgeX = smoothstep(0.0, edgeThickness, pos.x) * smoothstep(1.0, 1.0 - edgeThickness, pos.x);
float edgeY = smoothstep(0.0, edgeThickness, pos.y) * smoothstep(1.0, 1.0 - edgeThickness, pos.y);
float edge = min(edgeX, edgeY);
vec3 color = mix(mortarColor, brickColor, edge);
```

## optimization tips

### coherent branching
- avoid divergent flow control in fragment shaders
- consider replacing if/else with mathematical functions
- use discard sparingly

### precision qualifiers
- use appropriate precision for variables
- lowp, mediump, highp depending on needs
- lower precision = better performance

### shader complexity
- balance visual quality vs performance
- profile and identify performance bottlenecks
- consider alternative algorithms for expensive operations

### caching results
- calculate expensive operations once and reuse
- precompute values when possible
- use lookup textures for complex functions 