#include "mesh_utils.hpp"

#ifdef __EMSCRIPTEN__
    #include <GLES3/gl3.h>
#else
    #include <glad/glad.h>
#endif
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <tl/basic.hpp>

using glm::vec3;
constexpr float PI = glm::pi<float>();

namespace tg
{

void createScreenQuadMesh2D(u32& vao, u32& vbo, u32& numVerts)
{
    numVerts = 6;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(k_screenQuad2DVerts), k_screenQuad2DVerts, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
}

void createCubeMesh(u32& vao, u32& vbo, u32& numVerts, bool withNormals)
{
    numVerts = 36;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    if(withNormals)
        glBufferData(GL_ARRAY_BUFFER, sizeof(k_cubeVertsWithNormals), k_cubeVertsWithNormals, GL_STATIC_DRAW);
    else
        glBufferData(GL_ARRAY_BUFFER, sizeof(k_cubeVerts), k_cubeVerts, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6*sizeof(float), nullptr);
    if(withNormals) {
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6*sizeof(float), (void*)(3*sizeof(float)));
    }
}


/*static void generateIcosahedronVerts(vec3 verts[12])
{
    using glm::sqrt;
    using glm::mat2;
    constexpr float PI = glm::pi<float>();
    const mat2 rot36 = rotY(0.2f * PI);
    const float r = 1 / (2*sin(0.2f*PI));
    const float h = sqrt(1 - r*r);
    const float a = sqrt(r*r - 0.25f);
    const float h2 = 0.5f * sqrt(1 - (r-a)*(r-a) - 0.25f);
    verts[0] = vec3(0, h2+h, 0);
    vec2 p(r, 0);
    for(int i = 0; i < 5; i++) {
        verts[1+i] = {p.y, h2, -p.x};
        p = rot36 * p;
        verts[6+i] = {p.y, -h2, -p.x};
        p = rot36 * p;
    }
    verts[11] = -verts[0];
    for(int i = 0; i < 12; i++)
        verts[i] = glm::normalize(verts[i]);
}*/
static const vec3 s_icosahedronVerts[12] = {
    {0.0000000000000000000000000, 1.0000000000000000000000000, 0.0000000000000000000000000},
    {0.0000000000000000000000000, 0.4472136497497558593750000, -0.8944272398948669433593750},
    {-0.8506507873535156250000000, 0.4472136497497558593750000, -0.2763932347297668457031250},
    {-0.5257311463356018066406250, 0.4472136497497558593750000, 0.7236067652702331542968750},
    {0.5257310271263122558593750, 0.4472136497497558593750000, 0.7236068248748779296875000},
    {0.8506507873535156250000000, 0.4472136497497558593750000, -0.2763930857181549072265625},
    {-0.5257310867309570312500000, -0.4472136497497558593750000, -0.7236068248748779296875000},
    {-0.8506507873535156250000000, -0.4472136497497558593750000, 0.2763931453227996826171875},
    {-0.0000000626720364493849047, -0.4472136497497558593750000, 0.8944271802902221679687500},
    {0.8506507277488708496093750, -0.4472136497497558593750000, 0.2763932645320892333984375},
    {0.5257312059402465820312500, -0.4472136497497558593750000, -0.7236067056655883789062500},
    {-0.0000000000000000000000000, -1.0000000000000000000000000, -0.0000000000000000000000000},
};

void createIcoSphereMeshData(u32& numVerts, u32& numInds, vec3* verts, u32* inds, u32 subDivs)
{
    const int fnl = 1 << subDivs; // num levels per face
    numVerts =
        2 + // top and bot verts
        2 * 5 * (fnl-1) * fnl / 2 + // top and bot caps
        5 * (fnl+1) * fnl; // middle

    numInds = 3 * (20 * (1 << (2 * subDivs)));

    if(verts)
    {
        int vi = 0;
        auto addVert = [&](vec3 p) {
            verts[vi++] = glm::normalize(p);
        };
        addVert(s_icosahedronVerts[0]);

        using glm::mix;
        for(int l = 1; l < fnl; l++) {
            const float vertPercent = float(l) / fnl;
            for(int f = 0; f < 5; f++) {
                const vec3 pLeft = mix(s_icosahedronVerts[0], s_icosahedronVerts[1+f], vertPercent);
                const vec3 pRight = mix(s_icosahedronVerts[0], s_icosahedronVerts[1+(f+1)%5], vertPercent);
                for(int x = 0; x < l; x++) {
                    const vec3 p = mix(pLeft, pRight, float(x) / l);
                    addVert(p);
                }
            }
        }

        for(int l = 0 ; l < fnl; l++) {
            const float vertPercent = float(l) / fnl;
            for(int f = 0; f < 5; f++) {
                const vec3 topLeft = s_icosahedronVerts[1+f];
                const vec3 topRight = s_icosahedronVerts[1+(f+1)%5];
                const vec3 botLeft = s_icosahedronVerts[6+f];
                const vec3 botRight = s_icosahedronVerts[6+(f+1)%5];
                const vec3 left = mix(topLeft, botLeft, vertPercent);
                const vec3 mid = mix(topRight, botLeft, vertPercent);
                const vec3 right = mix(topRight, botRight, vertPercent);
                for(int x = 0; x < fnl-l; x++) {
                    const vec3 p = mix(left, mid, float(x) / (fnl-l));
                    addVert(p);
                }
                for(int x = 0; x < l; x++) {
                    const vec3 p = mix(mid, right, float(x) / l);
                    addVert(p);
                }
            }
        }

        for(int l = 0; l < fnl; l++) {
            const float vertPercent = float(l) / fnl;
            for(int f = 0; f < 5; f++) {
                const vec3 pLeft = mix(s_icosahedronVerts[6+f], s_icosahedronVerts[11], vertPercent);
                const vec3 pRight = mix(s_icosahedronVerts[6+(f+1)%5], s_icosahedronVerts[11], vertPercent);
                for(int x = 0; x < fnl-l; x++) {
                    const vec3 p = mix(pLeft, pRight, float(x) / (fnl-l));
                    addVert(p);
                }
            }
        }

        addVert(s_icosahedronVerts[11]);
        assert(vi == (int)numVerts);
    }

    if(inds)
    {
        int i = 0;
        auto addTri = [&](int i0, int i1, int i2) {
            inds[i++] = i0;
            inds[i++] = i1;
            inds[i++] = i2;
        };

        // top
        for(int f = 0; f < 5; f++)
            addTri(0, 1+f, 1+(f+1)%5);

        int rowOffset = 1;
        for(int l = 1; l < fnl; l++) {
            const int rowLen = l*5;
            const int nextRowLen = (l+1)*5;
            for(int f = 0; f < 5; f++) {
                for(int x = 0; x < l; x++) {
                    const int topLeft = rowOffset + l*f + x;
                    const int topRight = rowOffset + (l*f + x + 1) % rowLen;
                    const int botLeft = rowOffset + rowLen + (l+1) * f + x;
                    addTri(
                        topLeft,
                        botLeft,
                        botLeft + 1);
                    addTri(
                        topLeft,
                        botLeft + 1,
                        topRight);
                }
                addTri(
                    rowOffset + (l*(f+1)) % rowLen, // review!
                    rowOffset + rowLen + (l+1) * f + l,
                    rowOffset + rowLen + ((l+1) * f + l + 1) % nextRowLen);
            }
            rowOffset += rowLen;
        }

        { // middle
            const int rowLen = 5 * fnl;
            for(int l = 0; l < fnl; l++) {
                for(int x = 0; x < rowLen; x++) {
                    const int topLeft = rowOffset + x;
                    const int topRight = rowOffset + (x+1) % rowLen;
                    const int botLeft = rowOffset + rowLen + x;
                    const int botRight = rowOffset + rowLen + (x+1) % rowLen;
                    addTri(topLeft, botLeft, topRight);
                    addTri(topRight, botLeft, botRight);
                }
                rowOffset += rowLen;
            }
        }

        // bottom
        for(int l = 0; l < fnl-1; l++) {
            const int faceLen = fnl - l;
            const int rowLen = faceLen*5;
            const int nextRowLen = (faceLen-1)*5;
            for(int f = 0; f < 5; f++) {
                for(int x = 0; x < fnl-l-1; x++) {
                    const int topLeft = rowOffset + faceLen*f + x;
                    const int topRight = rowOffset + faceLen*f + x + 1;
                    const int botLeft = rowOffset + rowLen + (faceLen-1) * f + x;
                    const int botRight = rowOffset + rowLen + ((faceLen-1) * f + x + 1) % nextRowLen;
                    addTri(topLeft, botLeft, topRight);
                    addTri(topRight, botLeft, botRight);
                }
                addTri(
                    rowOffset + faceLen*(f+1) - 1,
                    rowOffset + rowLen + ((faceLen-1) * (f+1)) % nextRowLen,
                    rowOffset + faceLen*(f+1) % rowLen);
            }
            rowOffset += rowLen;
        }
        for(int f = 0; f < 5; f++)
            addTri(numVerts-6+f, numVerts-1, numVerts-6 + (f+1)%5);

        assert(i == (int)numInds);
    }
}

void createIcoSphereMesh(u32& vao, u32& vbo, u32& ebo, u32& numInds, u32 subDivs)
{
    u32 numVerts;
    createIcoSphereMeshData(numVerts, numInds, nullptr, nullptr, subDivs);
    vec3* positions = new vec3[numVerts];
    defer(delete[] positions);
    u32* inds = new u32[numInds];
    defer(delete[] inds);
    createIcoSphereMeshData(numVerts, numInds, positions, inds, subDivs);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, numVerts * sizeof(vec3), positions, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0); // positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(1); // normals
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    glGenBuffers(1, &ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, numInds * sizeof(u32), inds, GL_STATIC_DRAW);
}

void createCylinderMeshData(u32& numVerts, u32& numInds, Vert_pos_normal* verts, u32* inds,
    float radius, float minY, float maxY, u32 resolution)
{
    assert(resolution >= 3);
    numVerts = 4 * resolution;
    numInds = 2*3 * (resolution - 2) + resolution * 6;

    if(verts)
    {
        // bottom cap
        for(int i = 0; i < resolution; i++) {
            const float alpha = (2*PI * i) / resolution;
            const float x = sin(alpha);
            const float z = cos(alpha);
            verts[i] = {vec3{x, minY, z}, vec3{0, -1, 0}};
        }
        // top cap
        for(int i = 0; i < resolution; i++) {
            vec3 pos = verts[i].pos;
            pos.y = maxY;
            verts[resolution + i] = {pos, {0, +1, 0}};
        }
        // sides
        for(int i = 0; i < resolution; i++) {
            const vec3 normal = {verts[i].pos.x, 0, verts[i].pos.z};
            verts[2*resolution + 2*i + 0] = {verts[i].pos, normal};
            verts[2*resolution + 2*i + 1] = {verts[resolution+i].pos, normal};
        }
        for(int i = 0; i < 4*resolution; i++) {
            verts[i].pos.x *= radius;
            verts[i].pos.z *= radius;
        }
    }

    if(inds)
    {
        // bottom cap
        for(int i = 0; i < resolution-2; i++) {
            inds[3*i + 0] = 0;
            inds[3*i + 1] = i+2;
            inds[3*i + 2] = i+1;
        }
        // top cap
        int offset = 3 * (resolution-2);
        for(int i = 0; i < resolution-2; i++) {
            inds[offset + 3*i + 0] = resolution + 0;
            inds[offset + 3*i + 1] = resolution + i+1;
            inds[offset + 3*i + 2] = resolution + i+2;
        }
        // sides
        offset += offset;
        for(int i = 0; i < resolution; i++) {
            const int a0 = i;
            const int a1 = (i+1) % resolution;
            const int b0 = resolution + i;
            const int b1 = resolution + (i+1) % resolution;
            inds[offset + 6*i + 0] = a0;
            inds[offset + 6*i + 1] = a1;
            inds[offset + 6*i + 2] = b1;
            inds[offset + 6*i + 3] = a0;
            inds[offset + 6*i + 4] = b1;
            inds[offset + 6*i + 5] = b0;
        }
    }
}

void createCylinderMesh(u32& vao, u32& vbo, u32& ebo, u32& numInds,
    float radius, float minY, float maxY, u32 resolution)
{
    u32 numVerts;
    createCylinderMeshData(numVerts, numInds, nullptr, nullptr, radius, minY, maxY, resolution);
    auto verts = new Vert_pos_normal[numVerts];
    defer(delete[] verts);
    u32* inds = new u32[numInds];
    defer(delete[] inds);
    createCylinderMeshData(numVerts, numInds, verts, inds, radius, minY, maxY, resolution);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, numVerts * sizeof(Vert_pos_normal), verts, GL_STATIC_DRAW);
    const int stride = sizeof(Vert_pos_normal);
    glEnableVertexAttribArray(0); // positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, nullptr);
    glEnableVertexAttribArray(1); // normals
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, (void*)(sizeof(vec3)));

    glGenBuffers(1, &ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, numInds * sizeof(u32), inds, GL_STATIC_DRAW);
}

const float k_cubeVerts[6*6*3] = {
    // x, y, z,
    // LEFT
    -1, -1, -1,
    -1, -1, +1,
    -1, +1, +1,
    -1, -1, -1,
    -1, +1, +1,
    -1, +1, -1,
    // RIGHT
    +1, -1, +1,
    +1, -1, -1,
    +1, +1, -1,
    +1, -1, +1,
    +1, +1, -1,
    +1, +1, +1,
    // BOTTOM
    -1, -1, -1,
    +1, -1, -1,
    +1, -1, +1,
    -1, -1, -1,
    +1, -1, +1,
    -1, -1, +1,
    // TOP
    -1, +1, +1,
    +1, +1, +1,
    +1, +1, -1,
    -1, +1, +1,
    +1, +1, -1,
    -1, +1, -1,
    // FRONT
    -1, -1, +1,
    +1, -1, +1,
    +1, +1, +1,
    -1, -1, +1,
    +1, +1, +1,
    -1, +1, +1,
    // BACK
    +1, -1, -1,
    -1, -1, -1,
    -1, +1, -1,
    +1, -1, -1,
    -1, +1, -1,
    +1, +1, -1,
};

const float k_cubeVertsWithNormals[6*6*(3+3)] =
{
        // x, y, z,
        // LEFT
        -1, -1, -1,  -1, 0, 0,
        -1, -1, +1,  -1, 0, 0,
        -1, +1, +1,  -1, 0, 0,
        -1, -1, -1,  -1, 0, 0,
        -1, +1, +1,  -1, 0, 0,
        -1, +1, -1,  -1, 0, 0,
        // RIGHT
        +1, -1, +1,  +1, 0, 0,
        +1, -1, -1,  +1, 0, 0,
        +1, +1, -1,  +1, 0, 0,
        +1, -1, +1,  +1, 0, 0,
        +1, +1, -1,  +1, 0, 0,
        +1, +1, +1,  +1, 0, 0,
        // BOTTOM
        -1, -1, -1,  0, -1, 0,
        +1, -1, -1,  0, -1, 0,
        +1, -1, +1,  0, -1, 0,
        -1, -1, -1,  0, -1, 0,
        +1, -1, +1,  0, -1, 0,
        -1, -1, +1,  0, -1, 0,
        // TOP
        -1, +1, +1,  0, +1, 0,
        +1, +1, +1,  0, +1, 0,
        +1, +1, -1,  0, +1, 0,
        -1, +1, +1,  0, +1, 0,
        +1, +1, -1,  0, +1, 0,
        -1, +1, -1,  0, +1, 0,
        // FRONT
        -1, -1, +1,  0, 0, +1,
        +1, -1, +1,  0, 0, +1,
        +1, +1, +1,  0, 0, +1,
        -1, -1, +1,  0, 0, +1,
        +1, +1, +1,  0, 0, +1,
        -1, +1, +1,  0, 0, +1,
        // BACK
        +1, -1, -1,  0, 0, -1,
        -1, -1, -1,  0, 0, -1,
        -1, +1, -1,  0, 0, -1,
        +1, -1, -1,  0, 0, -1,
        -1, +1, -1,  0, 0, -1,
        +1, +1, -1,  0, 0, -1,
};

const float k_screenQuad2DVerts[6*2] = {
    -1, -1,  +1, -1,  +1, +1,
    -1, -1,  +1, +1,  -1, +1
};

}
