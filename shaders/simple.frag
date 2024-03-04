#version 450

layout (location = 0) in vec2 texCoord;

layout (location = 0) out vec4 color;

layout (binding = 0) uniform sampler2D colorTex;

void main() 
{
    color = textureLod(colorTex, texCoord, 0);
}