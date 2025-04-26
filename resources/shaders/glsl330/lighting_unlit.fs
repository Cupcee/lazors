#version 330

// Uniforms automatically uploaded by Raylib
uniform sampler2D texture0;   // the mesh’s diffuse texture (white by default)
uniform vec4      colDiffuse; // the material tint you set in Zig

out vec4 finalColor;

void main()
{
    // If your mesh has no useful texture you can drop `texture0` and
    // just write `finalColor = colDiffuse;`
    // Keeping the texture sample makes the shader work with *any* mesh.
    vec4 texel = texture(texture0, gl_PointCoord);   // or fragTexCoord if you keep it
    finalColor = texel * colDiffuse;
}

