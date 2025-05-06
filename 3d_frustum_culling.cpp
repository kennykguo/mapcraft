#include <3d_rendering.h>

// compute frustum planes for culling
// the view frustum is the 3D volume visible to the camera - truncated pyramid shape defined by six planes
// near plane - closest to the camera
// far plane - farthest from the camera
// left, top right, left - sides of the viewing volume
// several coordinate spaces:
// world space - global 3d coordinate system
// camera/view space - coordinates relative to the camera's position and orientation
// clip space - frustum becomes a cube
// normalized space - perspective division, final step for frustum culling
// world space -> view space -> clip space -> norm space -> screen space

void Renderer3D::compute_frustum_planes(glm::vec4* planes) {
    // compute view and projection matrices
    // the view matrix transforms world space to camera space using camera position and direction vector
    // projection matrix creates the perspective projection within a 45 deg view - 1024 x 768 aspect ratio
    // the near plane is at 0.1 units, the far plane is at 100000 units
    // internal variables defined in 3d_rendering.h

    glm::mat4 view = glm::lookAt(camera_pos, camera_pos + camera_front, camera_up);
    // camera_pos - position of camera in the world
    // camera_pos + camera_front - a point directly in front of the camera
    // camera_up - defines the direction of up in the camera

    // derive an orthonormal basis:
    // forward = -normalize(camera_pos + camera_front - camera_pos) = -normalize(camera_front)
    // right = normalize(cross(camera_up, forward))
    // up = cross(forward, right)

    // view matrix is constructed as:
    // [ right.x    right.y    right.z    -dot(right, camera_pos) ]
    // [ up.x       up.y       up.z       -dot(up, camera_pos)    ]
    // [ forward.x  forward.y  forward.z  -dot(forward, camera_pos)]
    // [ 0          0          0          1                        ]
    // this matrix is reponsible for rotating the world so the camera faces along the negative z-axis, and trnaslating the world so that the camera is at the origin

    // projection matrix maps a truncated pyramid to a cube in normalized device coordinates
    
    glm::mat4 proj = glm::perspective(glm::radians(45.0f), 1024.0f/768.0f, 0.1f, 10000.0f);
    // fov - vertical field of view angle
    // aspect ratio - width/height ratio
    // distance to near clipping plane
    // distance to far clipping plane

    // view matrix looks like:
    // [ f/aspect  0       0                   0                   ]
    // [ 0         f       0                   0                   ]
    // [ 0         0       (far+near)/(near-far)  (2*far*near)/(near-far) ]
    // [ 0         0       -1                  0                   ]
    // this matrix scales x and y based on fov and aspect ratio, maps z values between near and far to the range -1 and 1
    // introduces perspective division by setting the w component


    // calculate view-projection matrix
    // proj and view matrix are combined (matrix multiply)
    // this matrix is column-major
    // responsible for transforming from world space to clip space
    glm::mat4 vp = proj * view;
    // vp = | vp[0][0] vp[0][1] vp[0][2] vp[0][3] |
    //      | vp[1][0] vp[1][1] vp[1][2] vp[1][3] |
    //      | vp[2][0] vp[2][1] vp[2][2] vp[2][3] |
    //      | vp[3][0] vp[3][1] vp[3][2] vp[3][3] |

    // extract frustum planes from view-projection matrix
    // extracted by adding or subtracting specific columns of the vp matrix
    
    // each plane is a vec4 - x,y,z is the normal and w is the distance
    // think of this as the plane equation
    // for an obj to be visible, it must be inside all 6 planes

    // eg. left plane (column 4 + column 1)
    // standard form -  Ax + By + Cz + D = 0
    // can test a point is on one side of the plane by checking if:
    // Ax + By + Cz + D > 0 - positive side
    // Ax + By + Cz + D < 0 - negative side
    // planes are setup so that inside the frustum is the positive side
    // view matrix transforms into clip space used for rendering
    // in clip space, the equations correspond to specfic equations:

    // left plane: x ≥ -w
    // right plane: x ≤ w
    // bottom plane: y ≥ -w
    // top plane: y ≤ w
    // near plane: z ≥ -w
    // far plane: z ≤ w

    // eg.
    planes[0].x = vp[0][3] + vp[0][0];  // That's column 0, row 3 + column 0, row 0
    planes[0].y = vp[1][3] + vp[1][0];  // That's column 1, row 3 + column 1, row 0
    planes[0].z = vp[2][3] + vp[2][0];  // That's column 2, row 3 + column 2, row 0
    planes[0].w = vp[3][3] + vp[3][0];  // That's column 3, row 3 + column 3, row 0
    
    // right plane
    planes[1].x = vp[0][3] - vp[0][0]; // A
    planes[1].y = vp[1][3] - vp[1][0]; // B
    planes[1].z = vp[2][3] - vp[2][0]; // C
    planes[1].w = vp[3][3] - vp[3][0]; // D
    
    // bottom plane
    planes[2].x = vp[0][3] + vp[0][1];
    planes[2].y = vp[1][3] + vp[1][1];
    planes[2].z = vp[2][3] + vp[2][1];
    planes[2].w = vp[3][3] + vp[3][1];
    
    // top plane
    planes[3].x = vp[0][3] - vp[0][1];
    planes[3].y = vp[1][3] - vp[1][1];
    planes[3].z = vp[2][3] - vp[2][1];
    planes[3].w = vp[3][3] - vp[3][1];
    
    // near plane
    planes[4].x = vp[0][3] + vp[0][2];
    planes[4].y = vp[1][3] + vp[1][2];
    planes[4].z = vp[2][3] + vp[2][2];
    planes[4].w = vp[3][3] + vp[3][2];
    
    // far plane
    planes[5].x = vp[0][3] - vp[0][2];
    planes[5].y = vp[1][3] - vp[1][2];
    planes[5].z = vp[2][3] - vp[2][2];
    planes[5].w = vp[3][3] - vp[3][2];
    
    // normalize planes
    for (int i = 0; i < 6; i++) {
        float length = sqrt(planes[i].x * planes[i].x + 
                           planes[i].y * planes[i].y + 
                           planes[i].z * planes[i].z);
        planes[i] /= length;
    }

    // x' + w ≥ 0  (left)
    // -x' + w ≥ 0 (right)
    // y' + w ≥ 0  (bottom)
    // -y' + w ≥ 0 (top)
    // z' + w ≥ 0  (near)
    // -z' + w ≥ 0 (far)

    // transformation from world space to clip space - linear transformation of the following:
    // [x']   [vp[0][0] vp[0][1] vp[0][2] vp[0][3]]   [x]
    // [y'] = [vp[1][0] vp[1][1] vp[1][2] vp[1][3]] * [y]
    // [z']   [vp[2][0] vp[2][1] vp[2][2] vp[2][3]]   [z]
    // [w']   [vp[3][0] vp[3][1] vp[3][2] vp[3][3]]   [1]

    // by the properties of the clip space, we can check if a point is in the frustum by checking - 
    // [x']   [vp[0][0] vp[0][1] vp[0][2] vp[0][3]]   [x]
    // [y'] = [vp[1][0] vp[1][1] vp[1][2] vp[1][3]] * [y]
    // [z']   [vp[2][0] vp[2][1] vp[2][2] vp[2][3]]   [z]
    // [w']   [vp[3][0] vp[3][1] vp[3][2] vp[3][3]]   [1]

    // A = x
    // B = y
    // C = z
    // D = w
    // *** glm's matrix representation is in column major order. this means that the planes 
}


// determines which object is visible to the camera and only processes those objects
void Renderer3D::perform_frustum_culling() {
    // compute frustum planes
    glm::vec4 h_frustum_planes[6];
    compute_frustum_planes(h_frustum_planes);
    
    // allocate memory for frustum planes on GPU first time running
    if (!d_frustum_planes) {
        cuda_check(cudaMalloc(&d_frustum_planes, 6 * sizeof(float4)), "Allocate frustum planes");
    }
    
    // convert to CUDA float4 format and upload to GPU
    float4 h_cuda_planes[6];
    for (int i = 0; i < 6; i++) {
        h_cuda_planes[i].x = h_frustum_planes[i].x;
        h_cuda_planes[i].y = h_frustum_planes[i].y;
        h_cuda_planes[i].z = h_frustum_planes[i].z;
        h_cuda_planes[i].w = h_frustum_planes[i].w;
    }
    cuda_check(cudaMemcpy(d_frustum_planes, h_cuda_planes, 6 * sizeof(float4), cudaMemcpyHostToDevice), "Copy frustum planes to GPU");
    
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    // perform culling for buildings using spatial grid
    // check if we have a valid building total, valid cell range, device visible building indices not set to nullptr, and count not set to nullptr
    if (building_count > 0 && d_grid.cell_ranges && d_visible_building_indices && d_visible_building_count) {
        
        // reset visible count
        cuda_check(cudaMemset( d_visible_building_count, 0,  sizeof(int)), "Reset visible building count"
        );
        
        // launch kernel for frustum culling
        cuda_perform_frustum_culling( d_building_data, building_count, d_frustum_planes, d_grid, d_visible_building_indices, d_visible_building_count
        );
        cuda_check(cudaGetLastError(), "Launch building frustum culling kernel");
        
        // copy visible count back to host
        cuda_check(cudaMemcpy(&visible_building_count, d_visible_building_count, sizeof(int), cudaMemcpyDeviceToHost), "Copy visible building count to host");
    }
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    // road culling
    // future: implement spatial grid for roads too
    if (road_count > 0 && d_visible_road_indices && d_visible_road_count) {
        // reset visible count
        cuda_check(cudaMemset(d_visible_road_count, 0,  sizeof(int)),  "Reset visible road count" );
        
        // road culling kernel
        cuda_perform_road_culling( d_road_data, road_count, d_frustum_planes, d_visible_road_indices, d_visible_road_count);
        cuda_check(cudaGetLastError(), 
        "Launch road frustum culling kernel");
        
        // copy visible count back to host
        cuda_check(cudaMemcpy(&visible_road_count, d_visible_road_count, sizeof(int), cudaMemcpyDeviceToHost), "Copy visible road count to host");
    }

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    // perform culling for natural features
    if (natural_feature_count > 0 && d_visible_natural_feature_indices && d_visible_natural_feature_count) {
        // reset visible count before culling
        cuda_check(cudaMemset(d_visible_natural_feature_count, 0, sizeof(int)), "Reset visible natural feature count");
        
        // natural feature culling
        cuda_perform_natural_feature_culling(d_natural_feature_data, natural_feature_count, d_frustum_planes, d_visible_natural_feature_indices, d_visible_natural_feature_count
        );
        cuda_check(cudaGetLastError(), "Launch natural feature frustum culling kernel");
        
        // Copy visible count back to host
        cuda_check(cudaMemcpy(&visible_natural_feature_count, d_visible_natural_feature_count, sizeof(int), cudaMemcpyDeviceToHost), "Copy visible natural feature count to host");
    }
}