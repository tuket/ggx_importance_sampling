#include "shader_utils.hpp"

#include <stdio.h>
#include <assert.h>
#include <glad/glad.h>
#include <tl/basic.hpp>

namespace tg
{

char* getShaderCompileErrors(u32 shader, tl::Span<char> buffer)
{
    i32 ok;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if(!ok) {
        GLsizei outSize;
        glGetShaderInfoLog(shader, buffer.size(), &outSize, buffer.begin());
        return buffer.begin();
    }
    return nullptr;
}

char* getShaderLinkErrors(u32 prog, tl::Span<char> buffer)
{
    GLint success;
    glGetProgramiv(prog, GL_LINK_STATUS, &success);
    if(!success) {
        glGetProgramInfoLog(prog, buffer.size(), nullptr, buffer.begin());
        return buffer.begin();
    }
    return nullptr;
}

}