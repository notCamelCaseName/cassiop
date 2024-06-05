#version 450

layout(binding = 1) uniform FragUBO {
    float time;
} frag_ubo;

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 uv;

layout(location = 0) out vec4 outColor;

float sin_01(float x) {
    return 0.5 * sin(x) + 1.;
}

vec3 palette( float t ) {
    vec3 a = vec3(0.5, 0.5, 0.5);
    vec3 b = vec3(0.5, 0.5, 0.5);
    vec3 c = vec3(1.0, 1.0, 1.0);
    vec3 d = vec3(0.263,0.416,0.557);

    return a + b*cos( 6.28318*(c*t+d) );
}

void main() {
    vec2 uvb = uv;
    vec3 finalColor = vec3(0.0);

    for (float i = 0.0; i < 4.0; i++) {
        uvb = fract(uvb * 1.5) - 0.5;

        float d = length(uvb) * exp(-length(uv));

        vec3 col = palette(length(uv) + i*.4 + frag_ubo.time*.4);

        d = sin(d*8. + frag_ubo.time)/8.;
        d = abs(d);

        d = pow(0.01 / d, 1.2);

        finalColor += col * d;
    }

    outColor = vec4(finalColor, 1.0);
    //outColor = vec4(fragColor, 1.0);
}
