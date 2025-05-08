# OpenGL API Reference for MapCraft

This document provides a comprehensive overview of OpenGL concepts, functions, types, and usage patterns found in the MapCraft codebase.

## OpenGL Core Types

| Type | Description | Common Usage |
|------|-------------|--------------|
| `GLuint` | Unsigned integer | Object IDs (buffers, textures, shader programs) |
| `GLint` | Signed integer | Parameter values, uniform locations |
| `GLenum` | Enumeration constant | Mode selection, capability flags |
| `GLfloat` | Floating-point (32-bit) | Coordinates, colors, texture coordinates |
| `GLboolean` | Boolean value | Enable/disable flags |
| `GLsizei` | Signed integer for sizes | Array sizes, counts |

## OpenGL Objects

### Vertex Array Objects (VAO)

**Purpose**: Stores the configuration of vertex attribute data.

**Key Functions**:
```cpp
// Create VAO
GLuint vao;
glGenVertexArrays(1, &vao);

// Bind (make active)
glBindVertexArray(vao);

// Delete when done
glDeleteVertexArrays(1, &vao);
```

**Usage in MapCraft**:
- Building rendering: `building_vao`
- Terrain rendering: `terrain_vao`
- Car rendering: `car_vao`

### Vertex Buffer Objects (VBO)

**Purpose**: Stores vertex data (positions, normals, texture coordinates) in GPU memory.

**Key Functions**:
```cpp
// Create VBO
GLuint vbo;
glGenBuffers(1, &vbo);

// Bind for operations
glBindBuffer(GL_ARRAY_BUFFER, vbo);

// Upload data
glBufferData(GL_ARRAY_BUFFER, size_in_bytes, data_pointer, GL_STATIC_DRAW);

// Delete when done
glDeleteBuffers(1, &vbo);
```

**Usage Parameters**:
- `GL_STATIC_DRAW`: Data set once, used many times
- `GL_DYNAMIC_DRAW`: Data changed occasionally, used many times
- `GL_STREAM_DRAW`: Data changed every frame, used once

### Shader Programs

**Purpose**: Small GPU programs that process vertex and fragment data.

**Pipeline**:
1. Vertex Shader: Processes vertex positions and attributes
2. Tessellation Control Shader (optional): Controls tessellation levels
3. Tessellation Evaluation Shader (optional): Processes tessellated vertices
4. Geometry Shader (optional): Generates/modifies geometry primitives
5. Fragment Shader: Colors pixels

**Key Functions**:
```cpp
// Create shader
GLuint shader = glCreateShader(shader_type);
glShaderSource(shader, 1, &source, NULL);
glCompileShader(shader);

// Create program
GLuint program = glCreateProgram();
glAttachShader(program, vertex_shader);
glAttachShader(program, fragment_shader);
glLinkProgram(program);

// Use program
glUseProgram(program);
```

**Shader Types**:
- `GL_VERTEX_SHADER`: Processes vertices
- `GL_FRAGMENT_SHADER`: Processes fragments (pixels)
- `GL_TESS_CONTROL_SHADER`: Controls tessellation
- `GL_TESS_EVALUATION_SHADER`: Evaluates tessellated vertices
- `GL_GEOMETRY_SHADER`: Processes primitives

### Textures

**Purpose**: Stores image data for rendering.

**Key Functions**:
```cpp
// Create texture
GLuint texture;
glGenTextures(1, &texture);
glBindTexture(GL_TEXTURE_2D, texture);

// Set parameters
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

// Upload image data
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);

// Use texture in shader
glActiveTexture(GL_TEXTURE0);
glBindTexture(GL_TEXTURE_2D, texture);
glUniform1i(texLocation, 0);
```

**Texture Parameters**:
- Wrapping: `GL_REPEAT`, `GL_MIRRORED_REPEAT`, `GL_CLAMP_TO_EDGE`
- Filtering: `GL_NEAREST`, `GL_LINEAR`, `GL_LINEAR_MIPMAP_LINEAR`

## OpenGL Function Categories

### Attribute Setup

**Configuring Vertex Attributes**:
```cpp
// Define attribute layout
glVertexAttribPointer(
    0,                  // Attribute index/location
    3,                  // Number of components (x,y,z)
    GL_FLOAT,           // Data type
    GL_FALSE,           // Normalize data?
    3 * sizeof(float),  // Stride (bytes between vertices)
    (void*)0            // Offset to first component
);

// Enable attribute
glEnableVertexAttribArray(0);
```

**Common Attribute Indices**:
- 0: Position
- 1: Normal
- 2: Texture coordinates

### Uniform Variables

**Purpose**: Pass data to shaders that stays constant for all vertices/fragments.

**Key Functions**:
```cpp
// Get uniform location
GLint location = glGetUniformLocation(program, "modelMatrix");

// Set uniform values
glUniform1f(location, value);                      // Float
glUniform3fv(location, 1, glm::value_ptr(vector)); // Vec3
glUniformMatrix4fv(location, 1, GL_FALSE, glm::value_ptr(matrix)); // Mat4
```

**Common Uniforms in MapCraft**:
- Transformation matrices: model, view, projection
- Light properties: position, color
- Material properties: ambient, diffuse, specular, shininess

### Drawing

**Purpose**: Execute the rendering pipeline to draw primitives.

**Key Functions**:
```cpp
// Draw from array data
glDrawArrays(GL_TRIANGLES, 0, vertex_count);

// Draw indexed geometry
glDrawElements(GL_TRIANGLES, index_count, GL_UNSIGNED_INT, 0);
```

**Primitive Types**:
- `GL_POINTS`: Individual points
- `GL_LINES`: Line segments
- `GL_TRIANGLES`: Individual triangles
- `GL_TRIANGLE_STRIP`: Connected triangles
- `GL_TRIANGLE_FAN`: Triangles sharing a central vertex

### State Management

**Purpose**: Configure the OpenGL rendering pipeline behavior.

**Key Functions**:
```cpp
// Enable capabilities
glEnable(GL_DEPTH_TEST);   // Enable depth testing
glEnable(GL_BLEND);        // Enable alpha blending
glEnable(GL_CULL_FACE);    // Enable face culling

// Disable capabilities
glDisable(GL_DEPTH_TEST);

// Configure blending
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

// Configure depth testing
glDepthFunc(GL_LESS);
```

## Transformation Pipeline

**Model Matrix**: Positions object in world space (translation, rotation, scale)
```cpp
glm::mat4 model = glm::mat4(1.0f);                                  // Identity matrix
model = glm::translate(model, position);                            // Apply translation
model = glm::rotate(model, glm::radians(angle), glm::vec3(0,1,0));  // Apply rotation
model = glm::scale(model, glm::vec3(scale_x, scale_y, scale_z));    // Apply scaling
```

**View Matrix**: Positions camera in world space
```cpp
glm::mat4 view = glm::lookAt(
    camera_pos,             // Camera position
    camera_pos + camera_front, // Target position
    camera_up               // Up vector
);
```

**Projection Matrix**: Defines how 3D coordinates project to 2D screen
```cpp
// Perspective projection
glm::mat4 projection = glm::perspective(
    glm::radians(fov),       // Field of view
    aspect_ratio,           // Aspect ratio
    near_plane,             // Near clipping plane
    far_plane               // Far clipping plane
);

// Orthographic projection
glm::mat4 projection = glm::ortho(
    left, right,            // Left/right bounds
    bottom, top,            // Bottom/top bounds
    near_plane, far_plane   // Near/far planes
);
```

## Common Usage Patterns

### Complete Rendering Sequence

```cpp
// 1. Select shader program
glUseProgram(shader_program);

// 2. Set uniforms
glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));

// 3. Bind VAO (which has VBO configuration)
glBindVertexArray(vao);

// 4. Draw
glDrawArrays(GL_TRIANGLES, 0, vertex_count);

// 5. Unbind (optional but good practice)
glBindVertexArray(0);
```

### Creating and Uploading Mesh Data

```cpp
// 1. Generate buffers
GLuint vao, vbo_positions, vbo_normals;
glGenVertexArrays(1, &vao);
glGenBuffers(1, &vbo_positions);
glGenBuffers(1, &vbo_normals);

// 2. Bind VAO first
glBindVertexArray(vao);

// 3. Upload position data
glBindBuffer(GL_ARRAY_BUFFER, vbo_positions);
glBufferData(GL_ARRAY_BUFFER, positions.size() * sizeof(float), positions.data(), GL_STATIC_DRAW);
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
glEnableVertexAttribArray(0);

// 4. Upload normal data
glBindBuffer(GL_ARRAY_BUFFER, vbo_normals);
glBufferData(GL_ARRAY_BUFFER, normals.size() * sizeof(float), normals.data(), GL_STATIC_DRAW);
glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
glEnableVertexAttribArray(1);

// 5. Unbind VAO for cleanliness
glBindVertexArray(0);
```

## Key OpenGL State Functions

| Function | Purpose |
|----------|---------|
| `glEnable(GL_DEPTH_TEST)` | Enable depth testing to handle occlusion |
| `glDepthFunc(GL_LESS)` | Set depth comparison function |
| `glEnable(GL_BLEND)` | Enable alpha blending for transparency |
| `glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)` | Set standard alpha blending |
| `glEnable(GL_CULL_FACE)` | Enable face culling for performance |
| `glCullFace(GL_BACK)` | Cull back faces |
| `glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)` | Wireframe rendering mode |
| `glClearColor(r, g, b, a)` | Set color for clearing the screen |
| `glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)` | Clear color and depth buffers |

## Advanced Techniques in MapCraft

### Tessellation

Control the level of detail for meshes based on distance:

```cpp
// Tessellation control shader
layout (vertices = 3) out;

void main() {
    // Set tessellation levels
    gl_TessLevelOuter[0] = tessellation_level;
    gl_TessLevelOuter[1] = tessellation_level;
    gl_TessLevelOuter[2] = tessellation_level;
    gl_TessLevelInner[0] = tessellation_level;
}
```

### Shadow Mapping

Two-pass technique for realistic shadows:

1. **Pass 1**: Render scene from light's perspective to depth map
2. **Pass 2**: Render scene normally, using depth map to determine shadows

### Water Effects

- Wave animation using time-based displacement
- Reflections with environment mapping
- Refraction effects
- Alpha blending for transparency

## Debugging OpenGL

1. Check for errors:
```cpp
GLenum error = glGetError();
if (error != GL_NO_ERROR) {
    // Handle error
}
```

2. Use debug output (OpenGL 4.3+):
```cpp
glEnable(GL_DEBUG_OUTPUT);
glDebugMessageCallback(debugCallback, 0);
```

3. Validate shader programs:
```cpp
GLint success;
glGetProgramiv(program, GL_LINK_STATUS, &success);
if (!success) {
    char infoLog[512];
    glGetProgramInfoLog(program, 512, NULL, &infoLog);
    std::cerr << "Shader program linking failed: " << infoLog << std::endl;
}
``` 