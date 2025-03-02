#version 450

layout(binding = 0) uniform MVP {
    mat4 projection;
    mat4 view;
    mat4 model;
} mvp;

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
layout(location = 2) in vec2 uv;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 uv_out;

void main() {
    gl_Position = mvp.projection * mvp.view * mvp.model * vec4(position, 1.0);
    fragColor = color;
    uv_out = uv;
}
