#include "utils.hpp"
#include <tg/img.hpp>

static const char* k_vertShadSrc =
R"GLSL(
layout (location = 0) in vec2 a_pos;
out vec2 v_pos;

void main()
{
    v_pos = a_pos;
    gl_Position = vec4(a_pos, 0, 1.0);
}
)GLSL";

static const char* k_fragShadSrc =
R"GLSL(
layout (location = 0) out vec4 o_color;

in vec2 v_pos;

uniform float u_rough2;
uniform uint u_numSamples = 1024u;
uniform samplerCube u_envTex;
uniform float u_resolution; // resolution in each face of u_envTex
uniform mat3 u_rayBasis; // the basis frame for making rays

float DistributionGGX(vec3 N, vec3 H)
{
    float a2 = u_rough2*u_rough2;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;

    float nom   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return nom / denom;
}

float RadicalInverse_VdC(uint bits) 
{
     bits = (bits << 16u) | (bits >> 16u);
     bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
     bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
     bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
     bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
     return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}
vec2 Hammersley(uint i, uint N)
{
    return vec2(float(i)/float(N), RadicalInverse_VdC(i));
}

vec3 ImportanceSampleGGX(vec2 rand, vec3 N)
{
    float a = u_rough2;
    
    float phi = 2.0 * PI * rand.x;
    float cosTheta = sqrt((1.0 - rand.y) / (1.0 + (a*a - 1.0) * rand.y));
    float sinTheta = sqrt(1.0 - cosTheta*cosTheta);
    
    // from spherical coordinates to cartesian coordinates - halfway vector
    vec3 H;
    H.x = cos(phi) * sinTheta;
    H.y = sin(phi) * sinTheta;
    H.z = cosTheta;
    
    // from tangent-space H vector to world-space sample vector
    vec3 up        = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent   = normalize(cross(up, N));
    vec3 bitangent = cross(N, tangent);
    
    vec3 sampleVec = tangent * H.x + bitangent * H.y + N * H.z;
    return normalize(sampleVec);
}

void main()
{
    vec3 N = u_rayBasis * vec3(v_pos, 1);

    vec3 color = vec3(0.0);
    float w = 0.0;
    
    for(uint i = 0u; i < u_numSamples; ++i)
    {
        vec2 rand = Hammersley(i, u_numSamples);
        vec3 H = ImportanceSampleGGX(rand, N);
        float NdotH = max(dot(N, H), 0.0);
        vec3 L  = normalize(2.0 * NdotH * H - N);

        float NdotL = max(dot(N, L), 0.0);
        if(NdotL > 0.0)
        {
            float D = DistributionGGX(N, H);
            float pdf = D * NdotH / (4.0 * NdotH) + 0.0001; 

            float saTexel  = 4.0 * PI / (6.0 * u_resolution * u_resolution);
            float saSample = 1.0 / (float(u_numSamples) * pdf + 0.0001);

            float mipLevel = u_rough2 == 0.0 ? 0.0 : 0.5 * log2(saSample / saTexel); 
            
            color += textureLod(u_envTex, L, mipLevel).rgb * NdotL;
            w += NdotL;
        }
    }

    color = color / w;
    o_color = vec4(color, 1.0);
}
)GLSL";

int main(int argc, char** argv)
{
    glfwSetErrorCallback(+[](int error, const char* description) {
        fprintf(stderr, "Glfw Error %d: %s\n", error, description);
    });
    if (!glfwInit()) {
        fprintf(stderr, "error glfwInit\n");
        return 1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(1000, 800, "test ggx", nullptr, nullptr);
    if (window == nullptr) {
        fprintf(stderr, "error creating the window\n");
        return 2;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(0); // Enable vsync

    if (gladLoadGL() == 0) {
        fprintf(stderr, "error in gladLoadGL()\n");
        return 3;
    }
    glad_set_post_callback(glErrorCallback);

    glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);

    // init envmap texture
    u32 envmapTex;
    glGenTextures(1, &envmapTex);
    glBindTexture(GL_TEXTURE_CUBE_MAP, envmapTex);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    static const char* imgPath = "autumn_cube.hdr";
    tg::Img3f img(tg::Img3f::load(imgPath));
    if(img.data() == nullptr) {
        printf("error loading image file %s. The working dir must be the data/ folder \n", imgPath);
        return 5;
    }
    uploadCubemapTexture(0, img.width(), img.height(), GL_RGB16, GL_RGB, GL_FLOAT, (u8*)img.data());
    glGenerateMipmap(GL_TEXTURE_CUBE_MAP);

    // init FBO
    u32 fbo;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    GLenum fboDrawBuffers[] = {GL_COLOR_ATTACHMENT0};
    glDrawBuffers(1, fboDrawBuffers);

    // init screen quad
    u32 vao, vbo;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    static const float k_screenQuadVertData[] = { // intended to be drawn as GL_TRIANGLE_FAN
        -1,-1,  +1,-1,  +1,+1,  -1,+1
    };
    glBufferData(GL_ARRAY_BUFFER, sizeof(k_screenQuadVertData), k_screenQuadVertData, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);

    // init shader
    u32 prog = makeProgram(k_vertShadSrc, k_fragShadSrc);
    glUseProgram(prog);
    struct Locs {
        i32 rough2;
        i32 numSamples;
        i32 envTex;
        i32 resolution; // resolution in each face of u_envTex
        i32 rayBasis; // the basis frame for making rays
    };
    const Locs locs = {
        glGetUniformLocation(prog, "u_rough2"),
        glGetUniformLocation(prog, "u_numSamples"),
        glGetUniformLocation(prog, "u_envTex"),
        glGetUniformLocation(prog, "u_resolution"),
        glGetUniformLocation(prog, "u_rayBasis"),
    };

    const int numLevels = 10;
    for(int i = 0; i < numLevels; i++)
    {
        const int outW = img.width() / (1 << i);
        const int outH = img.height() / (1 << i);
        u32 outTexture;
        glGenTextures(1, &outTexture);
        glBindTexture(GL_TEXTURE_2D, outTexture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, outW, outW, 0, GL_RGBA, GL_FLOAT, nullptr);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, outTexture, 0);
        if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            printf("The framebuffer is not camplete\n");
            return 4;
        }
        defer(glDeleteTextures(1, &outTexture));


        for(int face = 0; face < 6; face++)
        {
            glUniform1f(locs.rough2, float(i) / (numLevels-1));
            glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
        }


        // save image!
    }
}