#pragma once

#include <tl/int_types.hpp>
#include <glm/vec3.hpp>

namespace tg
{

struct Vert_pos_normal { glm::vec3 pos, normal; };

void createScreenQuadMesh2D(u32& vao, u32& vbo, u32& numVerts);
void createCubeMesh(u32& vao, u32& vbo, u32& numVerts, bool withNormals);
// includes positions and normals (TRIANGLES)
void createIcoSphereMeshData(u32& numVerts, u32& numInds, glm::vec3* verts, u32* inds, u32 subDivs);
void createIcoSphereMesh(u32& vao, u32& vbo, u32& ebo, u32& numInds, u32 subDivs);

void createCylinderMeshData(u32& numVerts, u32& numInds, Vert_pos_normal* verts, u32* inds,
    float radius, float minY, float maxY, u32 resolution);
void createCylinderMesh(u32& vao, u32& vbo, u32& ebo, u32& numInds,
    float radius, float minY, float maxY, u32 resolution);

// vertex positions of a cube center in the origin of corrdinates with side of length 2
extern const float k_cubeVerts[6*6*3]; // positions only
extern const float k_cubeVertsWithNormals[6*6*(3+3)]; // postions and normals
// vertex positions(2D) for a quad to render in full screen
extern const float k_screenQuad2DVerts[6*2];
// vetex postions of an icosaedron
extern const float k_icosaedronVerts[20*3];
}
