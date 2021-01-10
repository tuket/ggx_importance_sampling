#include "utils.hpp"
#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>
#include <tg/img.hpp>
#include <tg/mesh_utils.hpp>
#include <tg/cameras.hpp>

using glm::vec3;
using glm::vec4;

// --- SHADERS ---
static const char k_vertShadSrc[] =
R"GLSL(
layout (location = 0) in vec3 a_pos;
layout (location = 1) in vec3 a_normal;

uniform mat4 u_model;
uniform mat4 u_modelViewProj;

out vec3 v_pos;
out vec3 v_normal;

void main()
{
    vec4 worldPos4 = u_model * vec4(a_pos, 1.0);
    v_pos = worldPos4.xyz / worldPos4.w;
    v_normal = (u_model * vec4(a_normal, 0.0)).xyz;
    gl_Position = u_modelViewProj * vec4(a_pos, 1.0);
}
)GLSL";

static const char k_rtFragShadSrc[] = // uses ray tracing to sample the environment
R"GLSL(
layout (location = 0) out vec4 o_uniform;
layout (location = 1) out vec4 o_importanceNDF;
layout (location = 2) out vec4 o_importanceVNDF;

uniform vec3 u_camPos;
uniform vec3 u_albedo;
uniform float u_rough2;
uniform float u_metallic;
uniform vec3 u_F0;
uniform samplerCube u_convolutedEnv;
uniform sampler2D u_lut;
uniform uint u_numSamples = 16u;
uniform uint u_numFramesWithoutChanging;

in vec3 v_pos;
in vec3 v_normal;

vec3 fresnelSchlick(float NoV, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(1.0 - NoV, 5.0);
}

float ggx_G(float NoV, float rough2)
{
  float rough4 = rough2 * rough2;
  return 2.0 * (NoV) /
    (NoV + sqrt(rough4 + (1.0 - rough4) * NoV*NoV));
}

float ggx_G_smith(float NoV, float NoL, float rough2)
{
  return ggx_G(NoV, rough2) * ggx_G(NoL, rough2);
}

// hash function for nerating psudorandom numbers
uvec3 pcg_uvec3_uvec3(uvec3 v)
{
    v = v * 1664525u + 1013904223u;
    v.x += v.y*v.z;
    v.y += v.z*v.x;
    v.z += v.x*v.y;
    v = v ^ (v>>16u);
    v.x += v.y*v.z;
    v.y += v.z*v.x;
    v.z += v.x*v.y;
    return v;
}

// Construct a float with half-open range [0:1] using low 23 bits.
// All zeroes yields 0.0, all ones yields the next smallest representable value below 1.0.
// https://stackoverflow.com/a/17479300/1754322
float makeFloat01( uint m ) {
    const uint ieeeMantissa = 0x007FFFFFu; // binary32 mantissa bitmask
    const uint ieeeOne      = 0x3F800000u; // 1.0 in IEEE binary32

    m &= ieeeMantissa;                     // Keep only mantissa bits (fractional part)
    m |= ieeeOne;                          // Add fractional part to 1.0

    float  f = uintBitsToFloat( m );       // Range [1:2]
    return f - 1.0;                        // Range [0:1]
}

vec3 uniformSample(vec2 seed, vec3 N)
{
    float phi = 2 * PI * seed.x;
    float r = sqrt(1 - seed.y*seed.y);
    vec3 v = vec3(r*cos(phi), seed.y, r*sin(phi));
    v = normalize(v);
    vec3 up = vec3(0, 1, 0);
    if(dot(up, N) > 0.99)
        up = vec3(1, 0, 0);
    vec3 X = normalize(cross(N, up));
    vec3 Z = cross(X, N);
    return X * v.x + N * v.y + Z * v.z;
}

vec3 importanceSampleGgxD(vec2 seed, float rough2, vec3 N)
{
    float phi = 2.0 * PI * seed.x;
    float cosTheta = sqrt((1.0 - seed.y) / (1 + (rough2*rough2 - 1) * seed.y));
    float sinTheta = sqrt(1.0 - cosTheta*cosTheta);
    vec3 h;
    h.x = sinTheta * cos(phi);
    h.y = cosTheta;
    h.z = sinTheta * sin(phi);
    vec3 up = abs(N.y) < 0.999 ? vec3(0, 1, 0) : vec3(1, 0, 0);
    vec3 tangentX = normalize(cross(up, N));
    vec3 tangentZ = cross(tangentX, N);
    return h.x * tangentX + h.y * N + h.z * tangentZ;
}

vec3 importanceSampleGgxVD(vec2 seed, float rough2, vec3 N, vec3 V)
{
    mat3 orthoBasis; // build an ortho normal basis with N as the up direction
    orthoBasis[1] = N;
    orthoBasis[0] = abs(dot(N, vec3(0, 0, 1))) < 0.01 ?
                        cross(N, vec3(1, 0, 0)) :
                        cross(N, vec3(0, 0, 1));
    orthoBasis[0] = normalize(orthoBasis[0]);
    orthoBasis[2] = cross(orthoBasis[0], N);

    mat3 invOrthoBasis = transpose(orthoBasis);
    V = invOrthoBasis * V;

    // stretch view
    V = normalize(vec3(rough2*V.x, V.y, rough2*V.z));
    // orthonormal basis
    vec3 T1 = (V.y < 0.999) ? normalize(cross(V, vec3(0,0,1))) : vec3(1,0,0);
    vec3 T2 = cross(T1, V);
    // sample point with polar coordinates (r, phi)
    float a = 1.0 / (1.0 + V.y);
    float r = sqrt(seed.x);
    float phi = (seed.y<a) ? seed.y/a * PI : PI + (seed.y-a)/(1.0-a) * PI;
    float P1 = r*cos(phi);
    float P2 = r*sin(phi)*((seed.y<a) ? 1.0 : V.y);
    // compute normal
    vec3 h = P1*T1 + P2*T2 + sqrt(max(0.0, 1.0 - P1*P1 - P2*P2))*V;
    // unstretch
    h = normalize(vec3(rough2*h.x, max(0.0, h.y), rough2*h.z));
    return orthoBasis * h;
}

vec3 visualizeSamples()
{
    vec3 N = normalize(v_normal);
    float c = 0;
    for(uint iSample = 0u; iSample < u_numSamples; iSample++)
    {
        uvec3 seedUInt = pcg_uvec3_uvec3(uvec3(1234u, 16465u, iSample));
        vec2 seed2 = vec2(makeFloat01(seedUInt.x), makeFloat01(seedUInt.y));
        vec3 L = uniformSample(seed2, vec3(0, 1, 0));
        c += pow(max(0, dot(L, N)), 5000);
    }
    //c /= u_numSamples;
    return vec3(min(1, c));
}

void calcLighting()
{
    vec3 N = normalize(v_normal);
    vec3 V = normalize(u_camPos - v_pos);
    float NoV = max(0.0001, dot(N, V));

    vec3 F0 = mix(vec3(0.04), u_albedo, u_metallic);
    vec3 F = fresnelSchlick(NoV, F0);
    vec3 k_spec = F;
    vec3 k_diff = (1 - k_spec) * (1 - u_metallic);

    // uniform sampling
    {
        vec3 color = vec3(0.0, 0.0, 0.0);
        for(uint iSample = 0u; iSample < u_numSamples; iSample++)
        {
            uint sampleId = iSample + u_numSamples * u_numFramesWithoutChanging;
            uvec3 seedUInt = pcg_uvec3_uvec3(uvec3(gl_FragCoord.x, gl_FragCoord.y, sampleId));
            vec2 seed2 = vec2(makeFloat01(seedUInt.x), makeFloat01(seedUInt.y));
            vec3 L = uniformSample(seed2, N);
            vec3 H = normalize(V + L);
            vec3 env = textureLod(u_convolutedEnv, L, 0.0).rgb;

            float NoL = max(0.0001, dot(N, L));
            float NoH = max(0.0001, dot(N, H));
            float NoH2 = NoH * NoH;
            float rough4 = u_rough2*u_rough2;
            float q = NoH2 * (rough4 - 1.0) + 1.0;
            float D = rough4 / (PI * q*q);
            float G = ggx_G_smith(NoV, NoL, u_rough2);
            vec3 fr = F * G * D / (4 * NoV * NoL);
            vec3 diffuse = k_diff * u_albedo / PI;
            color += (diffuse + fr) * env * NoL;
        }
        color *= 2*PI / float(u_numSamples);
        o_uniform = vec4(color, 1);
    }
    // importance sampling NDF
    {
        vec3 specular = vec3(0.0, 0.0, 0.0);
        for(uint iSample = 0u; iSample < u_numSamples; iSample++)
        {
            uint sampleId = iSample + u_numSamples * u_numFramesWithoutChanging;
            uvec3 seedUInt = pcg_uvec3_uvec3(uvec3(gl_FragCoord.x, gl_FragCoord.y, sampleId));
            seedUInt.xy += seedUInt.x;
            vec2 seed2 = vec2(makeFloat01(seedUInt.x), makeFloat01(seedUInt.y));
            vec3 H = importanceSampleGgxD(seed2, u_rough2, N);
            vec3 L = reflect(-V, H);
            if(dot(N, L) > 0) {
                vec3 env = textureLod(u_convolutedEnv, L, 0.0).rgb;
                float NoL = max(0.0001, dot(N, L));
                float NoH = max(0.0001, dot(N, H));
                float G = ggx_G_smith(NoV, NoL, u_rough2);
                specular += env * F*G* dot(L, H) / (NoL * NoH);
            }
        }
        specular /= float(u_numSamples);
        o_importanceNDF = vec4(specular, 1);
    }
    // importance sampling VNDF
    {
        vec3 specular = vec3(0.0, 0.0, 0.0);
        for(uint iSample = 0u; iSample < u_numSamples; iSample++)
        {
            uint sampleId = iSample + u_numSamples * u_numFramesWithoutChanging;
            uvec3 seedUInt = pcg_uvec3_uvec3(uvec3(gl_FragCoord.x, gl_FragCoord.y, sampleId));
            seedUInt.xy += seedUInt.x;
            vec2 seed2 = vec2(makeFloat01(seedUInt.x), makeFloat01(seedUInt.y));
            vec3 H = importanceSampleGgxVD(seed2, u_rough2, N, V);
            vec3 L = reflect(-V, H);
            if(dot(N, L) > 0) {
                vec3 env = textureLod(u_convolutedEnv, L, 0.0).rgb;
                float NoL = max(0.0001, dot(N, L));
                float NoH = max(0.0001, dot(N, H));
                specular += env * F * ggx_G(NoL, u_rough2);
            }
        }
        specular /= float(u_numSamples);
        o_importanceVNDF = vec4(specular, 1);
    }
}

void main()
{
    calcLighting();
    //o_color = vec4(visualizeSamples(), 1.0);
}
)GLSL";

// --- DATA ---
static GLFWwindow* s_window;

// splitter
static GLFWcursor* s_splitterCursor = nullptr;
static float s_splitterPercent = 0.5;
static bool s_draggingSplitter = false;

// orbit cam & mouse
static bool s_mousePressed = false;
static glm::vec2 s_prevMouse;
static OrbitCameraInfo s_orbitCam;

// FBO
static u32 s_fbo;
static u32 s_rt[3], s_depthRbo; // render textures (0: uniform sampling, 1: NDF sampling, 2: VNDF sampling)
static u32 s_screenQuadVbo, s_screenQuadVao;

// everonment cubemap
static u32 s_envmapTex;
static u32 s_envCubeVao, s_envCubeVbo;

// our sphere
static u32 s_objVao, s_objVbo, s_objEbo, s_objNumInds;

// shaders and uniforms
static u32 s_splatProg, s_envProg, s_rtProg;
static struct { i32 tex; } s_splatUnifLocs;
static struct { i32 modelViewProj, cubemap, gammaExp; } s_envShadUnifLocs;
struct CommonUnifLocs { i32 camPos, model, modelViewProj, albedo, rough2, metallic, F0, convolutedEnv, lut; };
static struct : public CommonUnifLocs {
    i32 numSamples, numFramesWithoutChanging;
} s_rtUnifLocs;

static int s_samplingMode[2] = {0, 1}; // 0: uniform sampling, 1: importance sampling NDF, 2: importance sampling VNDF
static float s_rough = 0.1f;
static u32 s_numSamplesPerFrame = 64;
static u32 s_numFramesWithoutChanging = 0; // the number of frames we have been drawing the exact same thing, we use this to compute the blenfing factor inorder to apply temporal antialiasing
static glm::vec3 s_albedo = {0.8f, 0.8f, 0.8f};
static float s_metallic = 0.99f;

static void drawGui()
{
    bool showDemo = false;
    if(showDemo)
        ImGui::ShowDemoWindow(&showDemo);

    ImGui::Begin("tweaks", 0, 0);
    {
        static const char* panelNames[2] = {"left", "right"};
        static const char* modeNames[3] = {"uniform", "NDF", "VNDF"};
        ImGui::PushItemWidth(80);
        ImGui::Text("Sampling mode: ");
        ImGui::SameLine();
        ImGui::Text("(?)");
        if(ImGui::IsItemHovered())
        {
            ImGui::BeginTooltip();
            ImGui::Text(
                "choose the sampling mode for the left/right panels:\n"
                "uniform sampling,\n"
                "importance sampling using the NDF (normal distibution function),\n"
                "importance sampling using the VNDF (visiable NDF)");
            ImGui::EndTooltip();
        }
        for (int i = 0; i < 2; i++)
        {
            if (i > 0) ImGui::SameLine();
            ImGui::PushID(i);
            ImGui::ListBox(panelNames[i], &s_samplingMode[i], modeNames, 3);
            ImGui::PopID();
        }
        ImGui::PopItemWidth();

        bool gottaRefresh = false;
        gottaRefresh |= ImGui::SliderFloat("Roughness", &s_rough, 0, 1.f, "%.5f", 1);
        
        int numSamples = s_numSamplesPerFrame;
        constexpr int maxSamples = 1024;
        ImGui::SliderInt("Samples per frame", &numSamples, 1, maxSamples);
        s_numSamplesPerFrame = tl::clamp(numSamples, 1, maxSamples);

        gottaRefresh |= ImGui::ColorEdit3("Albedo", &s_albedo[0]);
        gottaRefresh |= ImGui::SliderFloat("Metallic", &s_metallic, 0.f, 1.f);
        if(gottaRefresh)
            s_numFramesWithoutChanging = 0;
    }
    ImGui::End();
}

static bool hoveringSplitter(GLFWwindow* window)
{
    if (ImGui::GetIO().WantCaptureMouse)
        return false;
    int windowW, windowH;
    glfwGetWindowSize(window, &windowW, &windowH);
    const float splitterPixX = floorf(windowW * s_splitterPercent);
    return splitterPixX-2 <= s_prevMouse.x &&
           s_prevMouse.x <= splitterPixX+2;
}

int main()
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

    s_window = glfwCreateWindow(800, 600, "test ggx", nullptr, nullptr);
    if (s_window == nullptr) {
        fprintf(stderr, "error creating the window\n");
        return 2;
    }

    glfwMakeContextCurrent(s_window);
    glfwSwapInterval(0); // Enable vsync

    if (gladLoadGL() == 0) {
        fprintf(stderr, "error in gladLoadGL()\n");
        return 3;
    }
    glad_set_post_callback(glErrorCallback);

    int screenW, screenH;
    glfwGetFramebufferSize(s_window, &screenW, &screenH);

    s_splitterCursor = glfwCreateStandardCursor(GLFW_HRESIZE_CURSOR);
    defer(glfwDestroyCursor(s_splitterCursor));

    s_orbitCam.distance = 5;
    s_orbitCam.heading = 0;
    s_orbitCam.pitch = 0;

    glfwSetKeyCallback(s_window, [](GLFWwindow* window, int key, int /*scanCode*/, int action, int /*mods*/)
    {
        if(key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);
    });

    glfwSetMouseButtonCallback(s_window, [](GLFWwindow* window, int button, int action, int /*mods*/)
    {
        if (ImGui::GetIO().WantCaptureMouse) {
            s_mousePressed = false;
            s_draggingSplitter = false;
            return;
        }
        if(button == GLFW_MOUSE_BUTTON_1)
        {
            s_mousePressed = action == GLFW_PRESS;
            s_draggingSplitter = s_mousePressed && hoveringSplitter(window);
        }
    });

    glfwSetCursorPosCallback(s_window, [](GLFWwindow* window, double x, double y)
    {
        if(s_mousePressed) {
            int windowW, windowH;
            glfwGetWindowSize(window, &windowW, &windowH);
            if(s_draggingSplitter) {
                x = tl::clamp(x, 0., double(windowW));
                s_splitterPercent = x / double(windowW);
            }
            else {
                const glm::vec2 d = glm::vec2{x, y} - s_prevMouse;
                s_orbitCam.applyMouseDrag(d, {windowW, windowH});
                if(x != 0)
                    s_numFramesWithoutChanging = 0;
            }
        }
        const bool showSplitterCursor = s_draggingSplitter || hoveringSplitter(window);
        if(showSplitterCursor) {
            ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_NoMouseCursorChange;
            glfwSetCursor(window, s_splitterCursor);
        }
        else {
            ImGui::GetIO().ConfigFlags &= ~ImGuiConfigFlags_NoMouseCursorChange;
        }
        s_prevMouse = {x, y};
    });

    glfwSetScrollCallback(s_window, [](GLFWwindow* /*window*/, double /*dx*/, double dy)
    {
        if (ImGui::GetIO().WantCaptureMouse)
            return;
        s_orbitCam.applyMouseWheel(dy);
        if(dy != 0)
            s_numFramesWithoutChanging = 0;
    });

    glfwSetWindowSizeCallback(s_window, [](GLFWwindow* /*window*/, int width, int height)
    {
        s_numFramesWithoutChanging = 0;
        for(int i = 0; i < tl::size(s_rt); i++) {
            glBindTexture(GL_TEXTURE_2D, s_rt[i]);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
        }
        glBindRenderbuffer(GL_RENDERBUFFER, s_depthRbo);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height);
    });

    {
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO(); (void)io;
        //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
        //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

        // Setup Dear ImGui style
        ImGui::StyleColorsDark();
        //ImGui::StyleColorsClassic();

        // Setup Platform/Renderer bindings
        ImGui_ImplGlfw_InitForOpenGL(s_window, true);
        ImGui_ImplOpenGL3_Init();
    }

    glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);

    // init FBO
    glGenFramebuffers(1, &s_fbo);
    constexpr int numColorTargets = tl::size(s_rt);
    glGenTextures(numColorTargets, s_rt);
    glBindFramebuffer(GL_FRAMEBUFFER, s_fbo);
    for(int i = 0; i < numColorTargets; i++) {
        glBindTexture(GL_TEXTURE_2D, s_rt[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, screenW, screenH, 0, GL_RGBA, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, s_rt[i], 0);
    }
    {
        glGenRenderbuffers(1, &s_depthRbo);
        glBindRenderbuffer(GL_RENDERBUFFER, s_depthRbo);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, screenW, screenH);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, s_depthRbo);
    }
    GLenum fboDrawBuffers[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2};
    glDrawBuffers(tl::size(fboDrawBuffers), fboDrawBuffers);
    if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        printf("The framebuffer is not camplete\n");
        return 4;
    }

    { // init envmap texture
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_CUBE_MAP, s_envmapTex);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        static const char* imgPath = "low.hdr";
        tg::Img3f img(tg::Img3f::load(imgPath));
        if(img.data() == nullptr) {
            printf("error loading image file %s. The working dir must be the data/ folder \n", imgPath);
            return 5;
        }
        uploadCubemapTexture(0, img.width(), img.height(), GL_RGB16, GL_RGB, GL_FLOAT, (u8*)img.data());
        glGenerateMipmap(GL_TEXTURE_CUBE_MAP);
    }

    // init screen quad
    glGenVertexArrays(1, &s_screenQuadVao);
    glBindVertexArray(s_screenQuadVao);
    glGenBuffers(1, &s_screenQuadVbo);
    glBindBuffer(GL_ARRAY_BUFFER, s_screenQuadVbo);
    static const float k_screenQuadVertData[] = { // intended to be drawn as GL_TRIANGLE_FAN
        -1,-1,  +1,-1,  +1,+1,  -1,+1
    };
    glBufferData(GL_ARRAY_BUFFER, sizeof(k_screenQuadVertData), k_screenQuadVertData, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);

    // init environment cube mesh
    glGenVertexArrays(1, &s_envCubeVao);
    defer(glDeleteVertexArrays(1, &s_envCubeVao));
    glBindVertexArray(s_envCubeVao);
    glGenBuffers(1, &s_envCubeVbo);
    defer(glDeleteVertexArrays(1, &s_envCubeVbo));
    glBindBuffer(GL_ARRAY_BUFFER, s_envCubeVbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(tg::k_cubeVerts), tg::k_cubeVerts, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    // init object mesh
    tg::createIcoSphereMesh(s_objVao, s_objVbo, s_objEbo, s_objNumInds, 3);
    defer(
        glDeleteVertexArrays(1, &s_objVao);
        glDeleteBuffers(1, &s_objVbo);
        glDeleteBuffers(1, &s_objEbo);
    );

    // init shaders
    createSimpleCubemapShader(s_envProg,
        s_envShadUnifLocs.modelViewProj, s_envShadUnifLocs.cubemap, s_envShadUnifLocs.gammaExp);
    defer(glDeleteProgram(s_envProg));
    glUseProgram(s_envProg);
    glUniform1i(s_envShadUnifLocs.cubemap, 0);
    glUniform1f(s_envShadUnifLocs.gammaExp, 1.f / 2.2f);

    s_splatProg = glCreateProgram();
    {
        const u32 vertShad = glCreateShader(GL_VERTEX_SHADER);
        defer(glDeleteShader(vertShad));
        const char* vertSrcs[] = {k_headerShadSrc, k_splatVertShadSrc};
        glShaderSource(vertShad, 2, vertSrcs, nullptr);
        glCompileShader(vertShad);
        if(const char* errMsg = tg::getShaderCompileErrors(vertShad, g_buffer)) {
            tl::println("Error compiling vertex shader:\n", errMsg);
            return 1;
        }

        const u32 fragShad = glCreateShader(GL_FRAGMENT_SHADER);
        defer(glDeleteShader(fragShad));
        const char* fragSrcs[] = {k_headerShadSrc, k_splatFragShadSrc};
        glShaderSource(fragShad, 2, fragSrcs, nullptr);
        glCompileShader(fragShad);
        if(const char* errMsg = tg::getShaderCompileErrors(vertShad, g_buffer)) {
            tl::println("Error compiling fragment shader:\n", errMsg);
            return 1;
        }

        glAttachShader(s_splatProg, vertShad);
        glAttachShader(s_splatProg, fragShad);
        glLinkProgram(s_splatProg);
        if(const char* errMsg = tg::getShaderLinkErrors(s_splatProg, g_buffer)) {
            tl::println("Error linking program:\n", errMsg);
            return 1;
        }
        glDetachShader(s_splatProg, vertShad);
        glDetachShader(s_splatProg, fragShad);

        s_splatUnifLocs.tex = glGetUniformLocation(s_splatProg, "u_tex");
    }
    defer(glDeleteProgram(s_splatProg));

    s_rtProg = glCreateProgram();
    {
        const char* vertSrcs[] = { k_headerShadSrc, k_vertShadSrc };
        const char* rtFragSrcs[] = { k_headerShadSrc, k_rtFragShadSrc };
        constexpr int numVertSrcs = tl::size(vertSrcs);
        constexpr int numRtFragSrcs = tl::size(rtFragSrcs);
        int srcsSizes[tl::max(numVertSrcs, numRtFragSrcs)];

        u32 vertShad = glCreateShader(GL_VERTEX_SHADER);
        defer(glDeleteShader(vertShad));
        for(int i = 0; i < numVertSrcs; i++)
            srcsSizes[i] = strlen(vertSrcs[i]);
        glShaderSource(vertShad, numVertSrcs, vertSrcs, srcsSizes);
        glCompileShader(vertShad);
        if(const char* errMsg = tg::getShaderCompileErrors(vertShad, g_buffer)) {
            tl::println("Error compiling vertex shader:\n", errMsg);
            return 1;
        }

        u32 rtFragShad = glCreateShader(GL_FRAGMENT_SHADER);
        defer(glDeleteShader(rtFragShad));
        for(int i = 0; i < numRtFragSrcs; i++)
            srcsSizes[i] = strlen(rtFragSrcs[i]);
        glShaderSource(rtFragShad, numRtFragSrcs, rtFragSrcs, srcsSizes);
        glCompileShader(rtFragShad);
        if(const char* errMsg = tg::getShaderCompileErrors(rtFragShad, g_buffer)) {
            tl::println("Error compiling RT frament shader:\n", errMsg);
            return 1;
        }

        glAttachShader(s_rtProg, vertShad);
        glAttachShader(s_rtProg, rtFragShad);
        glLinkProgram(s_rtProg);
        if(const char* errMsg = tg::getShaderLinkErrors(s_rtProg, g_buffer)) {
            tl::println("Error compiling frament shader:\n", errMsg);
            return 1;
        }

        { // collect unif locs
            s_rtUnifLocs = {
                glGetUniformLocation(s_rtProg, "u_camPos"),
                glGetUniformLocation(s_rtProg, "u_model"),
                glGetUniformLocation(s_rtProg, "u_modelViewProj"),
                glGetUniformLocation(s_rtProg, "u_albedo"),
                glGetUniformLocation(s_rtProg, "u_rough2"),
                glGetUniformLocation(s_rtProg, "u_metallic"),
                glGetUniformLocation(s_rtProg, "u_F0"),
                glGetUniformLocation(s_rtProg, "u_convolutedEnv"),
                glGetUniformLocation(s_rtProg, "u_lut"),
            };
            s_rtUnifLocs.numSamples = glGetUniformLocation(s_rtProg, "u_numSamples");
            s_rtUnifLocs.numFramesWithoutChanging = glGetUniformLocation(s_rtProg, "u_numFramesWithoutChanging");
        }
    }
    defer(glDeleteProgram(s_rtProg));

    glEnable(GL_SCISSOR_TEST);

    while(!glfwWindowShouldClose(s_window))
    {
        glfwPollEvents();

        glfwGetFramebufferSize(s_window, &screenW, &screenH);
        glViewport(0, 0, screenW, screenH);

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        drawGui();

        const float aspectRatio = float(screenW) / screenH;
        int splitterX = screenW * s_splitterPercent;
        splitterX = tl::max(0, splitterX - 1);
        int splitterLineWidth = tl::min(1, screenW - splitterX);

        // calc matrices
        const glm::mat4 viewMtx = tg::calcOrbitCameraMtx(vec3(0, 0, 0), s_orbitCam.heading, s_orbitCam.pitch, s_orbitCam.distance);
        const glm::mat4 projMtx = glm::perspective(glm::radians(45.f), aspectRatio, 0.1f, 1000.f);
        const glm::mat4 viewProjMtx = projMtx * viewMtx;
        const glm::mat4 modelMtx(1);
        const glm::vec4 camPos4 = glm::affineInverse(viewMtx) * glm::vec4(0,0,0,1);

        // start drawing to render targets
        glBindFramebuffer(GL_FRAMEBUFFER, s_fbo);
        glEnable(GL_BLEND);
        glBlendFunc(GL_ONE_MINUS_CONSTANT_ALPHA, GL_CONSTANT_ALPHA); // this is for accumulating frames to make some sort of temporal antialiasing
        glBlendColor(0, 0, 0, float(s_numFramesWithoutChanging) / (1 + s_numFramesWithoutChanging));
        glScissor(0, 0, screenW, screenH);

        glEnable(GL_DEPTH_TEST);
        glDepthMask(GL_TRUE);
        glClear(GL_DEPTH_BUFFER_BIT);

        glEnable(GL_CULL_FACE);
        { // draw with uniform sampling and importance sampling
            if(s_numFramesWithoutChanging == 0) {
                glClearColor(0,0,0,0);
                glClear(GL_COLOR_BUFFER_BIT);
            }
            glUseProgram(s_rtProg);
            glUniform3fv(s_rtUnifLocs.camPos, 1, &camPos4[0]);
            glUniformMatrix4fv(s_rtUnifLocs.model, 1, GL_FALSE, &modelMtx[0][0]);
            glUniformMatrix4fv(s_rtUnifLocs.modelViewProj, 1, GL_FALSE, &viewProjMtx[0][0]);
            glUniform3fv(s_rtUnifLocs.albedo, 1, &s_albedo[0]);
            glUniform1f(s_rtUnifLocs.rough2, s_rough*s_rough);
            glUniform1f(s_rtUnifLocs.metallic, s_metallic);
            const glm::vec3 ironF0(0.56f, 0.57f, 0.58f);
            glUniform3fv(s_rtUnifLocs.F0, 1, &ironF0[0]);
            glUniform1i(s_rtUnifLocs.convolutedEnv, 1);
            glUniform1ui(s_rtUnifLocs.numSamples, s_numSamplesPerFrame);
            glUniform1ui(s_rtUnifLocs.numFramesWithoutChanging, s_numFramesWithoutChanging);
            glBindVertexArray(s_objVao);
            glDrawElements(GL_TRIANGLES, s_objNumInds, GL_UNSIGNED_INT, nullptr);
        }

        glDisable(GL_CULL_FACE);

        // now we start drawing to the screen
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        // draw background
        glDisable(GL_DEPTH_TEST); // no reading, no writing
        glDisable(GL_BLEND);
        {
            glm::mat4 viewMtxWithoutTranslation = viewMtx;
            viewMtxWithoutTranslation[3][0] = viewMtxWithoutTranslation[3][1] = viewMtxWithoutTranslation[3][2] = 0;
            const glm::mat4 viewProjMtx = projMtx * viewMtxWithoutTranslation;
            glUseProgram(s_envProg);
            glUniformMatrix4fv(s_envShadUnifLocs.modelViewProj, 1, GL_FALSE, &viewProjMtx[0][0]);
            glBindVertexArray(s_envCubeVao);
            glDrawArrays(GL_TRIANGLES, 0, 6*6);
        }
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        // draw two sides of the splitter
        glUseProgram(s_splatProg);
        glUniform1i(s_splatUnifLocs.tex, 0);
        glActiveTexture(GL_TEXTURE0);
        glBindVertexArray(s_screenQuadVao);
        // draw left side of the splitter
        glScissor(0, 0, splitterX, screenH);
        glBindTexture(GL_TEXTURE_2D, s_rt[s_samplingMode[0]]);
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
        // draw right side of the splitter
        glScissor(splitterX+splitterLineWidth, 0, screenW - (splitterX+splitterLineWidth), screenH);
        glBindTexture(GL_TEXTURE_2D, s_rt[s_samplingMode[1]]);
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

        // draw splitter line
        {
            glScissor(splitterX, 0, splitterLineWidth, screenH);
            glClearColor(0, 1, 0, 1);
            glClear(GL_COLOR_BUFFER_BIT);
        }

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(s_window);

        s_numFramesWithoutChanging++;
    }
}