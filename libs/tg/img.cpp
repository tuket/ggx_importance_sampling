#include "img.hpp"

#include <stb/stb_image.h>
#include <stb/stb_image_write.h>
#include <math.h>
#include <glm/glm.hpp>

using glm::vec3;
using glm::vec4;
using glm::u8vec3;
using glm::u8vec4;
using glm::u8;

namespace tg
{

static_assert(sizeof(glm::vec2) == 2*sizeof(float), "The compiler is adding some padding");
static_assert(sizeof(glm::vec3) == 3*sizeof(float), "The compiler is adding some padding");

template<typename T>
struct stbi_loadT;
template <>
struct stbi_loadT<float> { static constexpr auto fn = stbi_loadf; };
template <>
struct stbi_loadT<u8> { static constexpr auto fn = stbi_load; };

template <int NC, typename T>
Img<NC, T> Img<NC, T>::load(const char* fileName)
{
    typedef glm::vec<NC, T, glm::defaultp> vecTN;
    Img<NC, T> img;
    int nc;
    T* data = stbi_loadT<T>::fn(fileName, &img._w, &img._h, &nc, NC);
    img._stride = img._w;
    img._data = reinterpret_cast<vecTN*>(data);
    return img;
}
template Img<1, float> Img<1, float>::load(const char*);
template Img<3, float> Img<3, float>::load(const char*);
template Img<4, float> Img<4, float>::load(const char*);
template Img<1, u8> Img<1, u8>::load(const char*);
template Img<3, u8> Img<3, u8>::load(const char*);
template Img<4, u8> Img<4, u8>::load(const char*);

template <int NC>
static int saveImgFloatN(const char* fileName, const void* data, int w, int h, int stride, int quality)
{
    const char* ext = nullptr;
    for(const char* c = fileName; *c; c++)
        if(*c == '.')
            ext = c;
    if(!ext)
        return false;
    ext++;
    auto dataF = (const float*)data;
    u8* dataU8 = nullptr;
    if(ext[0] == 'b' || ext[0] == 'p' || ext[0] == 'j' || ext[0] == 't') {
        dataU8 = new u8[w*h*NC];
        const float invGamma = 1.f / 2.2f;
        int i = 0;
        for(int y = 0; y < h; y++)
        for(int x = 0; x < w; x++)
        for(int c = 0; c < NC; c++, i++)
            dataU8[i] = (u8)(255 * pow(dataF[i], invGamma));
    }
    int okay = 0;
    switch (ext[0])
    {
        case 'b':
            okay = stbi_write_bmp(fileName, w, h, NC, dataU8);
            break;
        case 'p':
            okay = stbi_write_png(fileName, w, h, NC, dataU8, stride*sizeof(u8)*NC);
            break;
        case 'j':
            okay = stbi_write_jpg(fileName, w, h, NC, dataU8, quality);
            break;
        case 't':
            okay = stbi_write_tga(fileName, w, h, NC, dataU8);
            break;
        case 'h':
            okay = stbi_write_hdr(fileName, w, h, NC, dataF);
            break;
    }
    delete[] dataU8;
    return okay;
}

template <>
bool Img<3, float>::save(const char* fileName, int quality)
{
    return saveImgFloatN<3>(fileName, _data, _w, _h, _stride, quality);
}

}
