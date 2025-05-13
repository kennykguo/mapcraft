# MapCraft: 3D Graphics Rendering Engine of OpenStreetMap Database
<img src="https://github.com/user-attachments/assets/20941666-3676-48fa-8216-44b1dca6db26" alt="67207864-5B5A-47B3-A3EE-9E8D63F9667F_1_105_c" />



https://github.com/user-attachments/assets/42e7bc21-38d1-4175-8c19-f5663a72ab97







## Overview
MapCraft is a high-performance 3D rendering engine for visualizing large-scale geographic and urban environments. It combines OpenGL rendering with CUDA acceleration to deliver real-time performance.

## Core Features
- **Terrain Engine**: Procedural terrain with height mapping and multi-texture blending
- **Water System**: Realistic water surfaces with waves, reflections, and refraction
- **Vegetation**: Dynamic tree placement with natural distribution algorithms
- **Urban Environment**: Buildings and road networks with detailed texturing
- **Advanced Lighting**: Phong model with real-time shadow mapping
- **Weather Effects**: Particle system for rain and atmospheric effects
- **CUDA Acceleration**: GPU-based frustum culling and spatial partitioning

## System Design

### Rendering System
- **Shader Pipeline**: Specialized shaders for terrain, water, buildings, shadows
- **Texture Management**: Procedural generation for natural and urban textures
- **Lighting Model**: Enhanced Phong lighting with ambient boosting and shadow reception
- **Mesh Generation**: Dynamic generation of terrain, water surfaces, and vegetation
- **Sky Rendering**: Gradient-based atmospheric simulation

### Accelerated Using
- **Spatial Grid**: Constant O(1) time complexity lookup through spatial partitioning for objects
- **Frustum Culling**: Parallel visibility determination for optimal rendering
- **Spatial Queries**: Thread-safe operations for dynamic scene management

### Key Components
- **Terrain System**: Height-mapped grid with multi-texture blending based on elevation
- **Water Renderer**: Gerstner wave simulation with transparency and light interactions
- **Vegetation Generator**: Natural distribution in greenspaces and along roads
- **Shadow Mapper**: Real-time shadow calculation with soft edges
- **Particle Engine**: Thousands of concurrent particles for weather effects
- **Building Renderer**: Customized building meshes with texturing and lighting

## Performance Optimizations
- GPU-accelerated visibility culling
- Spatial partitioning for scene queries
- Distance-based level of detail
- Shader-based animations
- Shared memory optimization in CUDA kernels
- Atomic operations for thread safety
- Memory management to prevent leaks
