# opengl api documentation for mapcraft

## core opengl types

| type | description | usage |
|------|-------------|-------|
| `gluint` | unsigned integer | object ids (buffers, textures, shader programs) |
| `glint` | signed integer | parameter values, uniform locations |
| `glenum` | enumeration constant | mode selection, capability flags |
| `glfloat` | 32-bit float | coordinates, colors, texture coordinates |
| `glboolean` | boolean value | enable/disable flags |
| `glsizei` | signed integer for sizes | array sizes, counts |

## core opengl objects

### vertex array objects (vao)
- container for vertex attribute configurations
- stores bindings between attributes and vbos
- creation - `glGenVertexArrays`
- binding - `glBindVertexArray`
- deletion - `glDeleteVertexArrays`

### vertex buffer objects (vbo)
- stores vertex data (positions, normals, uvs, etc)
- creation - `glGenBuffers`
- binding - `glBindBuffer(GL_ARRAY_BUFFER, vboId)`
- data transfer - `glBufferData`
- deletion - `glDeleteBuffers`

### element buffer objects (ebo)
- stores indices for indexed rendering
- binding - `glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, eboId)`
- enables efficient mesh rendering with shared vertices

### shader objects
- vertex shader - processes each vertex
- fragment shader - processes each fragment/pixel
- creation - `glCreateShader`
- source setting - `glShaderSource`
- compilation - `glCompileShader`
- info log - `glGetShaderInfoLog`
- deletion - `glDeleteShader`

### shader programs
- linking multiple shaders into usable program
- creation - `glCreateProgram`
- shader attachment - `glAttachShader`
- linking - `glLinkProgram`
- info log - `glGetProgramInfoLog`
- using - `glUseProgram`
- deletion - `glDeleteProgram`

### textures
- stores image data for surface details
- created with `glGenTextures`
- configured with `glTexParameteri`
- data uploaded with `glTexImage2D`
- example: `concrete_texture` for building walls

### framebuffers
- alternate render targets
- enables rendering to textures instead of screen
- created with `glGenFramebuffers`
- example: `shadow_map_fbo` for shadow mapping

## common opengl function patterns

### object creation
```cpp
// create vao
GLuint vao;
glGenVertexArrays(1, &vao);

// create buffer
GLuint buffer;
glGenBuffers(1, &buffer);

// create texture
GLuint texture;
glGenTextures(1, &texture);

// create framebuffer
GLuint framebuffer;
glGenFramebuffers(1, &framebuffer);

// create shader
GLuint shader = glCreateShader(GL_VERTEX_SHADER);

// create program
GLuint program = glCreateProgram();
```

### object binding
```cpp
// bind vao - stores vertex attribute configuration
glBindVertexArray(vao);

// bind buffer - selects buffer for operations
glBindBuffer(GL_ARRAY_BUFFER, buffer);

// bind texture - selects texture for operations
glBindTexture(GL_TEXTURE_2D, texture);

// bind framebuffer - redirects rendering to framebuffer
glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

// use shader program - activates shader for rendering
glUseProgram(program);
```

### data transfer
```cpp
// upload buffer data
glBufferData(GL_ARRAY_BUFFER, data_size, data_ptr, GL_STATIC_DRAW);

// upload texture data
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data_ptr);

// update subset of buffer
glBufferSubData(GL_ARRAY_BUFFER, offset, size, data_ptr);
```

### attribute configuration
```cpp
// specify vertex attribute layout
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, offset_ptr);

// enable vertex attribute
glEnableVertexAttribArray(0);
```

### uniform management
```cpp
// get uniform location
GLint location = glGetUniformLocation(program, "modelMatrix");

// set uniform values
glUniform1f(location, value);                          // float
glUniform3fv(location, 1, glm::value_ptr(vector));     // vec3
glUniformMatrix4fv(location, 1, GL_FALSE, glm::value_ptr(matrix)); // mat4
```

### rendering
```cpp
// draw arrays - renders primitives directly
glDrawArrays(GL_TRIANGLES, 0, vertex_count);

// draw elements - renders indexed primitives
glDrawElements(GL_TRIANGLES, index_count, GL_UNSIGNED_INT, 0);
```

### state management
```cpp
// enable capabilities
glEnable(GL_DEPTH_TEST);
glEnable(GL_CULL_FACE);

// disable capabilities
glDisable(GL_BLEND);

// set depth function
glDepthFunc(GL_LESS);

// set blend function
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

// set viewport dimensions
glViewport(0, 0, width, height);

// clear buffers
glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
```

## shader compilation process

### shader compilation workflow
1. create shader objects
2. attach source code
3. compile shaders
4. create program
5. attach shaders to program
6. link program
7. delete individual shaders
8. use program

```cpp
// example shader compilation workflow
GLuint vertex = glCreateShader(GL_VERTEX_SHADER);
glShaderSource(vertex, 1, &vertex_source, nullptr);
glCompileShader(vertex);

GLuint fragment = glCreateShader(GL_FRAGMENT_SHADER);
glShaderSource(fragment, 1, &fragment_source, nullptr);
glCompileShader(fragment);

GLuint program = glCreateProgram();
glAttachShader(program, vertex);
glAttachShader(program, fragment);
glLinkProgram(program);

glDeleteShader(vertex);
glDeleteShader(fragment);

glUseProgram(program);
```

### error checking
```cpp
// shader compilation error checking
GLint success;
glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
if (!success) {
    char info_log[512];
    glGetShaderInfoLog(shader, 512, nullptr, info_log);
    // handle error
}

// program linking error checking
glGetProgramiv(program, GL_LINK_STATUS, &success);
if (!success) {
    char info_log[512];
    glGetProgramInfoLog(program, 512, nullptr, info_log);
    // handle error
}
```

## advanced opengl techniques

### framebuffer operations
```cpp
// create framebuffer
GLuint fbo;
glGenFramebuffers(1, &fbo);
glBindFramebuffer(GL_FRAMEBUFFER, fbo);

// attach texture to framebuffer
glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);

// attach renderbuffer to framebuffer
glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo);

// check framebuffer status
if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
    // handle error
}

// return to default framebuffer
glBindFramebuffer(GL_FRAMEBUFFER, 0);
```

### instanced rendering
```cpp
// set up instanced attribute
glVertexAttribDivisor(attribute_index, 1);

// draw instanced
glDrawArraysInstanced(GL_TRIANGLES, 0, vertex_count, instance_count);
glDrawElementsInstanced(GL_TRIANGLES, index_count, GL_UNSIGNED_INT, 0, instance_count);
```

### vertex array objects workflow
```cpp
// create and configure vao
GLuint vao;
glGenVertexArrays(1, &vao);
glBindVertexArray(vao);

// configure vertex attributes (stored in vao)
glBindBuffer(GL_ARRAY_BUFFER, vbo_position);
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
glEnableVertexAttribArray(0);

glBindBuffer(GL_ARRAY_BUFFER, vbo_normal);
glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
glEnableVertexAttribArray(1);

glBindBuffer(GL_ARRAY_BUFFER, vbo_texcoord);
glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, 0);
glEnableVertexAttribArray(2);

// unbind vao for later use
glBindVertexArray(0);

// later, just bind vao to use all attributes
glBindVertexArray(vao);
glDrawArrays(GL_TRIANGLES, 0, vertex_count);
```

## glsl shader overview

### shader types
- vertex shader - processes vertex positions and attributes
- fragment shader - determines pixel colors
- geometry shader - creates/modifies primitives
- tessellation control shader - controls tessellation levels
- tessellation evaluation shader - processes tessellated vertices

### data types
- scalar: `float`, `int`, `bool`
- vectors: `vec2`, `vec3`, `vec4`, `ivec2`, `bvec4`, etc.
- matrices: `mat2`, `mat3`, `mat4`
- samplers: `sampler2D`, `samplerCube`, etc.
- user-defined structures using `struct`

### qualifiers
- `in` - input variable (from previous stage)
- `out` - output variable (to next stage)
- `uniform` - constant value from cpu
- `layout(location = X)` - explicit attribute location

### built-in variables
- `gl_Position` - vertex position output (vertex shader)
- `gl_FragCoord` - fragment screen position (fragment shader)
- `gl_PointSize` - point primitive size (vertex shader)
- `gl_VertexID` - current vertex index (vertex shader)
- `gl_FragColor` - (legacy) fragment output color

### basic shader structure
```glsl
// vertex shader
#version 330 core
layout(location = 0) in vec3 aPos;       // position attribute
layout(location = 1) in vec3 aNormal;    // normal attribute
layout(location = 2) in vec2 aTexCoord;  // texture coordinate attribute

out vec3 FragPos;      // output to fragment shader
out vec3 Normal;       // output to fragment shader
out vec2 TexCoord;     // output to fragment shader

uniform mat4 model;        // transformation matrices
uniform mat4 view;
uniform mat4 projection;

void main() {
    // transform position to world space for lighting
    FragPos = vec3(model * vec4(aPos, 1.0));
    
    // transform normal to world space
    Normal = mat3(transpose(inverse(model))) * aNormal;
    
    // pass texture coordinates to fragment shader
    TexCoord = aTexCoord;
    
    // set clip-space position
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}

// fragment shader
#version 330 core
in vec3 FragPos;       // input from vertex shader
in vec3 Normal;        // input from vertex shader
in vec2 TexCoord;      // input from vertex shader

out vec4 FragColor;    // output color

uniform vec3 viewPos;           // camera position
uniform vec3 lightPos;          // light position
uniform vec3 lightColor;        // light color
uniform vec3 objectColor;       // object base color
uniform sampler2D mainTexture;  // texture sampler

// material properties
uniform float ambient;
uniform float diffuse;
uniform float specular;
uniform float shininess;

void main() {
    // sample texture
    vec3 texColor = texture(mainTexture, TexCoord).rgb;
    
    // combine with object color
    vec3 baseColor = texColor * objectColor;
    
    // normalize vectors
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    
    // ambient component
    vec3 ambient_component = ambient * lightColor;
    
    // diffuse component
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse_component = diffuse * diff * lightColor;
    
    // specular component
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
    vec3 specular_component = specular * spec * lightColor;
    
    // combine components
    vec3 result = (ambient_component + diffuse_component + specular_component) * baseColor;
    FragColor = vec4(result, 1.0);
}
```

## key shader programs in mapcraft

### basic shader
- simplest shader for rendering without lighting
- uses model-view-projection transformation
- applies a single color to objects
- used for simple geometric elements

### phong lighting shader
- implements phong reflection model
- calculates ambient, diffuse, and specular lighting
- supports texturing and normal mapping
- configurable material properties
- handles lighting from directional light source
- used for most scene objects (buildings, terrain)

### tessellation shader
- implements dynamic level of detail
- controls subdivision of geometry at runtime
- uses tessellation control shader to set tessellation levels
- uses tessellation evaluation shader to place new vertices
- better detail for nearby objects while saving performance on distant ones

### water shader
- creates realistic water surface
- implements gerstner waves for natural wave motion
- calculates dynamic normals for lighting
- adds fresnel effect for reflectivity
- combines multiple wave patterns for complexity
- applies ripple effects and specular highlights

### shadow mapping shader
- creates depth maps from light's perspective
- consists of two parts:
  1. shadow map generation (depth-only rendering)
  2. shadow application (depth comparison)
- implements percentage-closer filtering for soft shadows
- handles bias to prevent shadow acne
- transforms world positions to light space for depth comparison

### terrain shader
- generates procedural terrain using perlin noise
- blends multiple textures based on height and slope
- mixes grass, rock, and soil textures
- applies small-scale detail variation
- receives shadows from shadow mapping system
- optimized with visibility culling

### particle shader
- handles rain particle rendering
- manipulates point sprites based on particle properties
- creates streak effects for falling rain
- uses alpha blending for transparency
- implements simple particle physics

## optimal rendering workflow

1. initialize opengl context and create window
2. load and compile shaders
3. load textures and other resources
4. create and configure vaos and vbos
5. setup scene data and upload to gpu
6. in render loop:
   - update view matrices based on camera
   - perform frustum culling to determine visible objects
   - render shadow maps if using shadows
   - bind appropriate shader for each object type
   - set shader uniforms (matrices, lighting, material)
   - bind vaos and textures
   - draw objects
   - swap buffers to display rendered frame
7. cleanup resources on exit

## common opengl state changes

- depth testing: `glEnable(GL_DEPTH_TEST)` - prevents objects behind others from drawing
- face culling: `glEnable(GL_CULL_FACE)` - skips rendering of back-facing triangles
- blending: `glEnable(GL_BLEND)` - enables transparency effects
- polygon offset: `glEnable(GL_POLYGON_OFFSET_FILL)` - prevents z-fighting
- wireframe: `glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)` - renders only edges

## best practices

- minimize state changes (sorting by shader, texture, etc.)
- use vaos for attribute configuration
- batch similar objects together
- use instancing for repeated geometry
- implement frustum culling to skip offscreen objects
- use indexed rendering for complex meshes
- release resources with appropriate delete functions
- check for errors regularly during development
- prefer explicit attribute locations in shaders
- use shader storage buffer objects for large data 

## opengl objects

### texture objects
- stores image data for sampling in shaders
- creation - `glGenTextures`
- binding - `glBindTexture(GL_TEXTURE_2D, textureId)`
- parameter setting - `glTexParameteri`
- data upload - `glTexImage2D`
- mipmap generation - `glGenerateMipmap`
- sampling in shader via samplers (sampler2D)

### framebuffer objects
- enables rendering to textures
- creation - `glGenFramebuffers`
- binding - `glBindFramebuffer`
- texture attachment - `glFramebufferTexture2D`
- renderbuffer attachment - `glFramebufferRenderbuffer`
- status check - `glCheckFramebufferStatus`
- cleanup - `glDeleteFramebuffers`

## data transfer patterns

### vertex attribute setup
```cpp
// position attribute
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (void*)0);
glEnableVertexAttribArray(0);

// normal attribute
glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, (void*)(3 * sizeof(float)));
glEnableVertexAttribArray(1);

// texture coordinate attribute
glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, (void*)(6 * sizeof(float)));
glEnableVertexAttribArray(2);
```

### uniform setting
- floats - `glUniform1f`, `glUniform2f`, `glUniform3f`, `glUniform4f`
- integers - `glUniform1i`, `glUniform2i`, `glUniform3i`, `glUniform4i`
- matrices - `glUniformMatrix4fv`
- get location - `glGetUniformLocation(program, "uniformName")`

## shader compilation process
```cpp
// create and compile vertex shader
unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
glCompileShader(vertexShader);

// create and compile fragment shader
unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
glCompileShader(fragmentShader);

// link shaders into program
unsigned int shaderProgram = glCreateProgram();
glAttachShader(shaderProgram, vertexShader);
glAttachShader(shaderProgram, fragmentShader);
glLinkProgram(shaderProgram);

// cleanup
glDeleteShader(vertexShader);
glDeleteShader(fragmentShader);
```

## framebuffer operations
- creating fbo for shadow map:
```cpp
unsigned int depthMapFBO;
glGenFramebuffers(1, &depthMapFBO);

unsigned int depthMap;
glGenTextures(1, &depthMap);
glBindTexture(GL_TEXTURE_2D, depthMap);
glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, 
             SHADOW_WIDTH, SHADOW_HEIGHT, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthMap, 0);
glDrawBuffer(GL_NONE);
glReadBuffer(GL_NONE);
glBindFramebuffer(GL_FRAMEBUFFER, 0);
```

## optimal rendering workflow
1. clear buffers
   ```cpp
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
   ```

2. prepare camera view-projection matrix
   ```cpp
   glm::mat4 view = camera.getViewMatrix();
   glm::mat4 projection = glm::perspective(glm::radians(camera.fov), aspect, 0.1f, 100.0f);
   ```

3. bind shader program
   ```cpp
   glUseProgram(shaderProgram);
   ```

4. set global uniforms
   ```cpp
   glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
   glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));
   ```

5. frustum culling check
   ```cpp
   if (!isInFrustum(object.getBoundingBox(), camera.getFrustum()))
       continue;
   ```

6. bind vao and textures
   ```cpp
   glBindVertexArray(object.vao);
   glActiveTexture(GL_TEXTURE0);
   glBindTexture(GL_TEXTURE_2D, diffuseMap);
   ```

7. draw call
   ```cpp
   glDrawElements(GL_TRIANGLES, object.indexCount, GL_UNSIGNED_INT, 0);
   ```

8. unbind objects
   ```cpp
   glBindVertexArray(0);
   glUseProgram(0);
   ```

## best practices

### state management
- minimize state changes
- batch similar objects together
- update uniforms only when needed
- sort by shader, then texture, then mesh

### error handling
```cpp
GLenum err;
while((err = glGetError()) != GL_NO_ERROR) {
    // log error code
}
```

### memory management
- delete all created objects when done
- use vao and vbo object managers
- destroy textures and shaders when no longer needed

### debugging
- use debug context and callbacks
```cpp
glEnable(GL_DEBUG_OUTPUT);
glDebugMessageCallback(debugCallback, 0);
```

### performance
- use instanced rendering for repeated objects
- use indirect drawing for dynamic batches
- consider persistent mapped buffers for frequent updates
