#include <stdio.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <tl/basic.hpp>
#include <tl/fmt.hpp>
#include <tg/shader_utils.hpp>

constexpr float PI = glm::pi<float>();

constexpr int BUFFER_SIZE = 4*1024;
static char g_buffer[BUFFER_SIZE];

static const char* geGlErrStr(GLenum const err)
{
    switch (err) {
    case GL_NO_ERROR: return "GL_NO_ERROR";
    case GL_INVALID_ENUM: return "GL_INVALID_ENUM";
    case GL_INVALID_VALUE: return "GL_INVALID_VALUE";
    case GL_INVALID_OPERATION: return "GL_INVALID_OPERATION";
    case GL_INVALID_FRAMEBUFFER_OPERATION: return "GL_INVALID_FRAMEBUFFER_OPERATION";
    case GL_OUT_OF_MEMORY: return "GL_OUT_OF_MEMORY";
#ifdef GL_STACK_UNDERFLOW
    case GL_STACK_UNDERFLOW: return "GL_STACK_UNDERFLOW";
#endif
#ifdef GL_STACK_OVERFLOW
    case GL_STACK_OVERFLOW: return "GL_STACK_OVERFLOW";
#endif
    default:
        assert(!"unknown error");
        return nullptr;
    }
}

static void glErrorCallback(const char *name, void *funcptr, int len_args, ...) {
    GLenum error_code;
    error_code = glad_glGetError();
    if (error_code != GL_NO_ERROR) {
        fprintf(stderr, "ERROR %s in %s\n", geGlErrStr(error_code), name);
        assert(false);
    }
}


struct OrbitCameraInfo {
    float heading, pitch;
    float distance;
    void applyMouseDrag(glm::vec2 deltaPixels, glm::vec2 screenSize);
    void applyMouseWheel(float dy);
};

inline void OrbitCameraInfo::applyMouseDrag(glm::vec2 deltaPixels, glm::vec2 screenSize)
{
    constexpr float speed = PI;
    heading -= speed * deltaPixels.x / screenSize.x;
    while(heading < 0)
        heading += 2*PI;
    while(heading > 2*PI)
        heading -= 2*PI;
    pitch -= speed * deltaPixels.y / screenSize.y;
    pitch = glm::clamp(pitch, -0.45f*PI, +0.45f*PI);
}

inline void OrbitCameraInfo::applyMouseWheel(float dy)
{
    constexpr float speed = 1.04f;
    distance *= pow(speed, (float)dy);
    distance = glm::max(distance, 0.01f);
}

static u8 getNumChannels(u32 format)
{
    switch(format)
    {
    case GL_RED:
        return 1;
    case GL_RG:
        return 2;
    case GL_RGB:
    case GL_BGR:
            return 3;
    case GL_RGBA:
    case GL_BGRA:
        return 4;
    // There are many more but I'm lazy
    }
    assert(false);
    return 0;
}

static u8 getGetPixelSize(u32 format, u32 type)
{
    const u32 nc = getNumChannels(format);
    switch(type)
    {
    case GL_UNSIGNED_BYTE:
    case GL_BYTE:
        return nc;
    case GL_UNSIGNED_SHORT:
    case GL_SHORT:
    case GL_HALF_FLOAT:
        return 2*nc;
    case GL_UNSIGNED_INT:
    case GL_INT:
    case GL_FLOAT:
        return 4*nc;
    // there are many more but I'm lazy
    }
    assert(false);
    return 1;
}

static void uploadCubemapTexture(u32 mipLevel, u32 w, u32 h, u32 internalFormat, u32 dataFormat, u32 dataType, u8* data)
{
    const u8 ps = getGetPixelSize(dataFormat, dataType);
    const u32 side = w / 4;
    assert(3*side == h);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, w);
    defer(glPixelStorei(GL_UNPACK_ROW_LENGTH, 0));
    auto upload = [&](GLenum face, u32 offset) {
        glTexImage2D(face, mipLevel, internalFormat, side, side, 0, dataFormat, dataType, data + offset);
    };
    upload(GL_TEXTURE_CUBE_MAP_NEGATIVE_X, ps * w*side);
    upload(GL_TEXTURE_CUBE_MAP_POSITIVE_X, ps * (w*side + 2*side));
    upload(GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, ps * (w*2*side + side));
    upload(GL_TEXTURE_CUBE_MAP_POSITIVE_Y, ps * side);
    upload(GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, ps * (w*side + 3*side));
    upload(GL_TEXTURE_CUBE_MAP_POSITIVE_Z, ps * (w*side + side));
}

const char* k_headerShadSrc =
R"GLSL(
#version 330 core

const float PI = 3.14159265359;
)GLSL";

static const char k_simpleCubemapVertShadSrc[] =
R"GLSL(
#version 330 core

layout(location = 0) in vec3 a_pos;

out vec3 v_modelPos;

uniform mat4 u_modelViewProjMat;

void main()
{
    v_modelPos = a_pos;
    gl_Position = u_modelViewProjMat * vec4(a_pos, 1);
}
)GLSL";

static const char k_simpleCubemapFragShadSrc[] =
R"GLSL(
#version 330 core

layout(location = 0) out vec4 o_color;

in vec3 v_modelPos;

uniform samplerCube u_cubemap;
uniform float u_gammaExponent = 1.0;

void main()
{
    o_color = texture(u_cubemap, normalize(v_modelPos));
    o_color.rbg = pow(o_color.rbg, vec3(u_gammaExponent));
}
)GLSL";

void createSimpleCubemapShader(u32& prog,
    i32& modelViewProjUnifLoc, i32& cubemapTexUnifLoc, i32& gammaExpUnifLoc)
{
    prog = glCreateProgram();

    u32 vertShad = glCreateShader(GL_VERTEX_SHADER);
    defer(glDeleteShader(vertShad));
    const char* vertShadSrc = k_simpleCubemapVertShadSrc;
    glShaderSource(vertShad, 1, &vertShadSrc, nullptr);
    glCompileShader(vertShad);
    if(const char* errorMsg = tg::getShaderCompileErrors(vertShad, g_buffer)) {
        printf("Error compiling vertex shader:\n%s\n", errorMsg);
        assert(false);
    }

    u32 fragShad = glCreateShader(GL_FRAGMENT_SHADER);
    defer(glDeleteShader(fragShad));
    const char* fragShadSrc = k_simpleCubemapFragShadSrc;
    glShaderSource(fragShad, 1, &fragShadSrc, nullptr);
    glCompileShader(fragShad);
    if(const char* errorMsg = tg::getShaderCompileErrors(fragShad, g_buffer)) {
        printf("Error compiling fragment shader:\n%s\n", errorMsg);
        assert(false);
    }

    glAttachShader(prog, vertShad);
    glAttachShader(prog, fragShad);
    glLinkProgram(prog);
    if(const char* errorMsg = tg::getShaderLinkErrors(prog, g_buffer)) {
        printf("Error linking:\n%s\n", errorMsg);
        assert(false);
    }

    modelViewProjUnifLoc = glGetUniformLocation(prog, "u_modelViewProjMat");
    cubemapTexUnifLoc = glGetUniformLocation(prog, "u_cubemap");
    gammaExpUnifLoc = glGetUniformLocation(prog, "u_gammaExponent");
}

static const char k_splatVertShadSrc[] =
R"GLSL(
layout (location = 0) in vec2 a_pos;

out vec2 v_tc;

void main()
{
    v_tc = 0.5 + 0.5 * a_pos ;
    gl_Position = vec4(a_pos, 0, 1);
}
)GLSL";

static const char k_splatFragShadSrc[] =
R"GLSL(
layout (location = 0) out vec4 o_color;

in vec2 v_tc;

uniform sampler2D u_tex;

void main()
{
    vec4 color = texture(u_tex, v_tc);
    o_color = vec4(pow(color.rgb, vec3(1/2.2)), color.a);
}
)GLSL";

static void printShader(const char** srcs, int numSrcs)
{
    int line = 1;
    for(int srcInd = 0; srcInd < numSrcs; srcInd++) {
        const char* src = srcs[srcInd];
        const int n = strlen(src);
        int lineStart = 0;
        int i = 0;
        while(i < n) {
            while(i < n && src[i] != '\n')
                i++;
            
            printf("%3d | %.*s\n", line, i-lineStart, src+lineStart);
            lineStart = i+1;
            i++;
            line++;
        }
    }
}

static u32 makeShader(tl::Span<const char*> srcs, u32 type)
{
    const char* hsrcs[32];
    hsrcs[0] = k_headerShadSrc;
    for(int i = 0; i < srcs.size(); i++)
        hsrcs[i+1] = srcs[i];
    const u32 shader = glCreateShader(type);
    glShaderSource(shader, srcs.size()+1, hsrcs, nullptr);
    glCompileShader(shader);
    i32 ok;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if(!ok) {
        glGetShaderInfoLog(shader, BUFFER_SIZE, nullptr, g_buffer);
        printf("%s\n", g_buffer);
        printShader(hsrcs, srcs.size()+1);
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

static u32 makeProgram(tl::Span<const char*> vertSrc, tl::Span<const char*> fragSrc)
{
    const u32 vertShad = makeShader(vertSrc, GL_VERTEX_SHADER);
    if(vertShad == 0)
        return 0;

    const u32 fragShad = makeShader(fragSrc, GL_FRAGMENT_SHADER);
    if(fragShad == 0)
        return 0;

    const u32 prog = glCreateProgram();
    glAttachShader(prog, vertShad);
    glAttachShader(prog, fragShad);

    glLinkProgram(prog);

    i32 ok;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if(!ok) {
        glGetProgramInfoLog(prog, BUFFER_SIZE, nullptr, g_buffer);
        printf("Link Error:\n%s\n", g_buffer);
        glDetachShader(prog, vertShad);
        glDetachShader(prog, fragShad);
        glDeleteShader(vertShad);
        glDeleteShader(fragShad);
        return 0;
    }

    return prog;
}

static u32 makeProgram(u32 vertShad, tl::Span<const char*> fragSrc)
{
    const u32 fragShad = makeShader(fragSrc, GL_FRAGMENT_SHADER);
    if(fragShad == 0)
        return 0;

    const u32 prog = glCreateProgram();
    glAttachShader(prog, vertShad);
    glAttachShader(prog, fragShad);

    glLinkProgram(prog);

    i32 ok;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if(!ok) {
        glGetProgramInfoLog(prog, BUFFER_SIZE, nullptr, g_buffer);
        printf("Link Error:\n%s\n", g_buffer);
        glDetachShader(prog, fragShad);
        glDeleteShader(fragShad);
        return 0;
    }

    return prog;
}