#version 460


layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(gl_FragCoord.xy, 0.0, 1.0);
}
