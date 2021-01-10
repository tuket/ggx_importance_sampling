#pragma once

#include <tl/int_types.hpp>
#include <tl/span.hpp>

namespace tg
{

char* getShaderCompileErrors(u32 shad, tl::Span<char> buffer);
char* getShaderLinkErrors(u32 prog, tl::Span<char> buffer);

}
