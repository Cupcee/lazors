#version 330

// Per-vertex attributes coming from the mesh
in vec3 vertexPosition;

// Per-instance transform (mat4 uses 4 consecutive attribute slots)
in mat4 instanceTransform;

// Uniforms that Raylib automatically uploads
uniform mat4 mvp;

// We don’t need to send anything to the fragment shader
// so there are no out variables.

void main()
{
    // Just transform the vertex by the per-instance matrix
    // and the camera’s mvp matrix.
    gl_Position = mvp * instanceTransform * vec4(vertexPosition, 1.0);
}

