#include "utils.hpp"
#include <tg/img.hpp>
#include <stb/stb_image_write.h>

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

static const char* k_fragShadCommonSrc =
R"GLSL(
uniform float u_rough2;
uniform uint u_numSamples = 10u*1024u;

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
)GLSL";

static const char* k_cubemapShadSrc =
R"GLSL(
layout (location = 0) out vec4 o_color;

in vec2 v_pos;

uniform float u_resolution; // resolution in each face of u_envTex
uniform mat3 u_rayBasis; // the basis frame for making rays
uniform samplerCube u_envTex;

void main()
{
    vec3 N = u_rayBasis * vec3(v_pos, 1);
    N = normalize(N);

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
            float pdf = 0.25 * D; 

            float saTexel  = 4.0 * PI / (6.0 * u_resolution * u_resolution); // Solid Angle Texel (4*PI: whole sphere radians, 6*w*w: num pixels in the whole cubemap)
            float saSample = 1.0 / (float(u_numSamples) * pdf + 0.0001); // Solid Angle Sample

            float mipLevel = u_rough2 == 0.0 ? 0.0 : 0.5 * log2(saSample / saTexel); 
            
            color += textureLod(u_envTex, L, mipLevel).rgb * NdotL;
            w += NdotL;
        }
    }

    color = color / w;
    o_color = vec4(color, 1.0);
}
)GLSL";

static const char* k_latlongShadSrc =
R"GLSL(
layout (location = 0) out vec4 o_color;

in vec2 v_pos;

uniform float u_resolution; // width*height is of the mip 0
uniform sampler2D u_envTex;

vec2 dirToLatlongTc(vec3 v)
{
    return 0.5 + 0.5 * vec2(
        -atan(v.x, -v.z) / PI,
        asin(v.y) / (0.5*PI)
    );
}
vec3 latlongPosToDir(vec2 tc)
{
    float phi = tc.x * PI;
    float theta = 0.5*PI*tc.y;
    float l = cos(theta);
    return vec3(
        -sin(phi) * l,
        sin(theta),
        -cos(phi) * l
    );
}

void main()
{
    vec3 N = latlongPosToDir(v_pos);
    N = normalize(N);

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

            float saTexel  = 4.0 * PI / u_resolution;
            float saSample = 1.0 / (float(u_numSamples) * pdf + 0.0001);

            float mipLevel = u_rough2 == 0.0 ? 0.0 : 0.5 * log2(saSample / saTexel); 
            
            vec2 L2 = dirToLatlongTc(L);
            color += textureLod(u_envTex, L2, mipLevel).rgb * NdotL;
            w += NdotL;
        }
    }

    color = color / w;
    o_color = vec4(color, 1.0);
}
)GLSL";

// picture: http://www.reindelsoftware.com/Documents/Mapping/drawings/cubemap_uvxy.gif
const glm::mat3 rayBases[] = {
    glm::mat3({+1, 0, 0}, {0, +1, 0}, {0, 0, -1}), // front
    glm::mat3({0, 0, -1}, {0, +1, 0}, {-1, 0, 0}), // left
    glm::mat3({0, 0, +1}, {0, +1, 0}, {+1, 0, 0}), // left
    glm::mat3({-1, 0, 0}, {0, +1, 0}, {0, 0, +1}), // back
    glm::mat3({+1, 0, 0}, {0, 0, -1}, {0, -1, 0}), // down
    glm::mat3({+1, 0, 0}, {0, 0, +1}, {0, +1, 0}), // up
};
const glm::ivec2 fboOffsets[] = {
    {1, 1}, // front
    {0, 1}, // left
    {2, 1}, // right
    {3, 1}, // back
    {1, 0}, // down
    {1, 2}, // up
};
u32 fbo;

int prefilterCubemap(const char* fileName, const char* outFilePrefix)
{
	// init envmap texture
	u32 envmapTex;
	glGenTextures(1, &envmapTex);
	glBindTexture(GL_TEXTURE_CUBE_MAP, envmapTex);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	tg::Img3f img(tg::Img3f::load(fileName));
	if (img.data() == nullptr) {
		printf("error loading image file %s. The working dir must be the data/ folder \n", fileName);
		return 5;
	}
	uploadCubemapTexture(0, img.width(), img.height(), GL_RGB32F, GL_RGB, GL_FLOAT, (u8*)img.data());
	glGenerateMipmap(GL_TEXTURE_CUBE_MAP);

	// init shader
	const char* fragSrcs[2] = {k_fragShadCommonSrc, k_cubemapShadSrc};
	const u32 prog = makeProgram({&k_vertShadSrc, 1}, fragSrcs);
	glUseProgram(prog);
    if (prog == 0)
        return 6;
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

	int faceSize = img.width() / 4;
	int numLevels = 0;
	while (faceSize) {
		numLevels++;
		faceSize >>= 1;
	}
	faceSize = img.width() / 4;
	float* outPixels = new float[3 * 4 * faceSize * 3 * faceSize];
	u32 outTexture;
	glGenTextures(1, &outTexture);
	defer(glDeleteTextures(1, &outTexture));

	for (int i = 0; i < numLevels; i++)
	{
		glBindTexture(GL_TEXTURE_2D, outTexture);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 4 * faceSize, 3 * faceSize, 0, GL_RGBA, GL_FLOAT, nullptr);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, outTexture, 0);
		if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
			printf("The framebuffer is not complete\n");
			return 4;
		}
		const float rough = float(i) / (numLevels - 1);
		const float rough2 = rough * rough;
		glUniform1f(locs.rough2, rough2);
		glUniform1i(locs.envTex, 0);
		glBindTexture(GL_TEXTURE_CUBE_MAP, envmapTex);
		glUniform1f(locs.resolution, float(faceSize));

		for (int face = 0; face < 6; face++) {
			glUniformMatrix3fv(locs.rayBasis, 1, GL_FALSE, &rayBases[face][0][0]);
			const int tx0 = faceSize * fboOffsets[face].x;
			const int ty0 = faceSize * fboOffsets[face].y;
			glViewport(tx0, ty0, faceSize, faceSize);
			glScissor(tx0, ty0, faceSize, faceSize);
			glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
		}

		// save image!
		glFinish();
		glBindTexture(GL_TEXTURE_2D, outTexture);
		glReadPixels(0, 0, 4 * faceSize, 3 * faceSize, GL_RGB, GL_FLOAT, outPixels);
        char outFileName[128];
        tl::toStringBuffer(outFileName, outFilePrefix, i, ".hdr");
		stbi_write_hdr(outFileName, 4*faceSize, 3*faceSize, 3, outPixels);

		faceSize >>= 1;
	}

    return 0;
}

int prefilterLatlong(const char* fileName, const char* outFilePrefix)
{
	// init envmap texture
	u32 envmapTex;
	glGenTextures(1, &envmapTex);
	glBindTexture(GL_TEXTURE_2D, envmapTex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	tg::Img3f img(tg::Img3f::load(fileName));
	if (img.data() == nullptr) {
		printf("error loading image file %s. The working dir must be the data/ folder \n", fileName);
		return 5;
	}
	const int W = img.width();
	const int H = img.height();
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, W, H, 0, GL_RGB, GL_FLOAT, (u8*)img.data());
	glGenerateMipmap(GL_TEXTURE_2D);

    const char* fragSrcs[2] = {k_fragShadCommonSrc, k_latlongShadSrc};
	const u32 prog = makeProgram({&k_vertShadSrc, 1}, fragSrcs);
	glUseProgram(prog);
    if (prog == 0)
        return 6;
	struct Locs {
		i32 rough2;
		i32 numSamples;
		i32 envTex;
		i32 resolution; // width*height
	};
	const Locs locs = {
		glGetUniformLocation(prog, "u_rough2"),
		glGetUniformLocation(prog, "u_numSamples"),
		glGetUniformLocation(prog, "u_envTex"),
		glGetUniformLocation(prog, "u_resolution"),
	};

	float* outPixels = new float[3 * W*H];
    int w = W;
    int h = H;
	int numLevels = 0;
	while (h > 2) {
		numLevels++;
        h >>= 1;
	}
    h = H;
	u32 outTexture;
	glGenTextures(1, &outTexture);
	defer(glDeleteTextures(1, &outTexture));

	for (int i = 0; i < numLevels; i++)
	{
		glBindTexture(GL_TEXTURE_2D, outTexture);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0, GL_RGBA, GL_FLOAT, nullptr);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, outTexture, 0);
		if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
			printf("The framebuffer is not complete\n");
			return 4;
		}

		const float rough = float(i) / (numLevels - 1);
		const float rough2 = rough * rough;
		glUniform1f(locs.rough2, rough2);
		glUniform1i(locs.envTex, 0);
		glBindTexture(GL_TEXTURE_2D, envmapTex);
		glUniform1f(locs.resolution, float(W*H));
		glViewport(0, 0, w, h);
		glScissor(0, 0, w, h);
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

		// save image!
		glFinish();
		glBindTexture(GL_TEXTURE_2D, outTexture);
		glReadPixels(0, 0, w, h, GL_RGB, GL_FLOAT, outPixels);
		char outFileName[128];
		tl::toStringBuffer(outFileName, outFilePrefix, i, ".hdr");
		stbi_write_hdr(outFileName, w, h, 3, outPixels);

		w >>= 1;
        h >>= 1;
	}

	return 0;
}

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

    GLFWwindow* window = glfwCreateWindow(4, 4, "prefiltering...", nullptr, nullptr);
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

    // init FBO
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

    if (1)
        prefilterCubemap("autumn_cube.hdr", "autumn_cube_");
    else
        prefilterLatlong("estadio.hdr", "estadio_");
}