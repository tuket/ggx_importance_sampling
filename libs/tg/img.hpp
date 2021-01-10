#pragma once

#include <stdlib.h>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

namespace tg
{

template <int NC, typename T>
class ImgView
{
public:
    using VecT = glm::vec<NC, T>;
    using ImgViewConst = ImgView<NC, const T>;
    ImgView() : _w(0), _h(0), _stride(0), _data(nullptr) {}
    ImgView(int w, int h, VecT* data) : _w(w), _h(h), _stride(w), _data(data) {}
    ImgView(int w, int h, int stride, VecT* data) : _w(w), _h(h), _stride(stride), _data(data) {}
    operator ImgViewConst()const { return ImgView(_w, _h, _data); }

    const VecT& operator()(int x, int y)const;
    VecT& operator()(int x, int y);

    int width()const { return _w; }
    int height()const { return _h; }
    int stride()const { return _stride; }
    int strideInBytes()const { return _stride * sizeof(VecT); }
    constexpr int pixelSize()const { return sizeof(VecT); }
    VecT* data() { return _data; }
    const VecT* data()const { return _data; }
    int dataSize()const { return _w * _h * pixelSize(); }

    ImgView subImg(int x, int y, int w, int h);
    ImgViewConst subImb(int x, int y, int w, int h)const;

protected:
    int _w, _h;
    int _stride; // this stride is in elements, not in bytes. Stride is the number of elements from one row to next one
    VecT* _data;
};

template <int NC, typename T>
class Img : public ImgView<NC, T>
{
public:
    using VecT = glm::vec<NC, T, glm::defaultp>;
    using ImgViewT = ImgView<NC, T>;
    Img() : ImgViewT() {}
    Img(int w, int h) : ImgViewT(w, h, (T*)malloc(sizeof(T)*w*h)) {}
    Img(Img&& o);
    Img& operator=(Img&& o);
    ~Img() { free(ImgViewT::_data); }

    static Img load(const char* fileName);
    bool save(const char* fileName, int quality = 90); // quality is only used for jpeg and should be in range [1, 100]
};

enum class ECubeImgFace { LEFT, RIGHT, DOWN, UP, FRONT, BACK };

template <int NC, typename T>
struct CubeImgView
{
    using VecT = glm::vec<NC, T, glm::defaultp>;
    using ImgViewT = ImgView<NC, T>;
    using ImgViewConstT = ImgView<NC, const T>;
    struct Face { VecT* data; int stride; };
    int sidePixels;
    Face left, right, down, up, front, back;

    // create from an image view, specifiying the botton-left coordinate of each face in pixels (order of faces specified in ECubeImgFace)
    static CubeImgView createFromSingleImg(ImgViewT imgView, int sidePixels, const glm::ivec2 (&faceCoords)[6]);
    ImgViewT operator[](ECubeImgFace eFace);
    ImgViewConstT operator[](ECubeImgFace eFace)const;
};

template <int NC, typename T>
using CImg = ImgView<NC, const T>;
template <int NC, typename T>
using CCubeImg = CubeImgView<NC, const T>;

typedef Img<1, float> Img1f;
typedef Img<3, float> Img3f;
typedef Img<4, float> Img4f;
typedef Img<1, uint8_t> Img1u8;
typedef Img<3, uint8_t> Img3u8;
typedef Img<4, uint8_t> Img4u8;

typedef ImgView<3, float> ImgView3f;
typedef ImgView<4, float> ImgView4f;
typedef ImgView<3, uint8_t> ImgView3u8;
typedef ImgView<4, uint8_t> ImgView4u8;
typedef CubeImgView<3, float> CubeImgView3f;
typedef CubeImgView<3, float> CubeImgView4f;
typedef CubeImgView<3, uint8_t> CubeImgView3u8;
typedef CubeImgView<4, uint8_t> CubeImgView4u8;

typedef CImg<3, float> CImg3f;
typedef CImg<4, float> CImg4f;
typedef CImg<3, uint8_t> CImg3u8;
typedef CImg<4, uint8_t> CImg4u8;
typedef CCubeImg<3, float> CCubeImg3f;
typedef CCubeImg<4, float> CCubeImg4f;
typedef CCubeImg<3, uint8_t> CCubeImg3u8;
typedef CCubeImg<4, uint8_t> CCubeImg4u8;

// --- impl -------------------------------------------------------------------------------------------------------

template <int NC, typename T>
const glm::vec<NC, T, glm::defaultp>& ImgView<NC, T>::operator()(int x, int y)const {
    assert(x >= 0 && x < _w && y >= 0 && y < _h);
    return _data[x + _stride*y];
}
template <int NC, typename T>
glm::vec<NC, T, glm::defaultp>& ImgView<NC, T>::operator()(int x, int y) {
    assert(x >= 0 && x < _w && y >= 0 && y < _h);
    return _data[x + _stride*y];
}

template <int NC, typename T>
ImgView<NC, T> ImgView<NC, T>::subImg(int x, int y, int w, int h) {
    ImgView<NC, T> sub(w, h, _stride, _data + x + y*_stride);
    return sub;
}
template <int NC, typename T>
ImgView<NC, const T> ImgView<NC, T>::subImb(int x, int y, int w, int h)const {
    ImgView<NC, const T> sub(w, h, _stride, _data + x + y*_stride);
    return sub;
}

template <int NC, typename T>
CubeImgView<NC, T> CubeImgView<NC, T>::createFromSingleImg(ImgView<NC, T> imgView, int faceSide, const glm::ivec2 (&faceCoords)[6])
{
    CubeImgView<NC, T> view;
    view.sidePixels = faceSide;
    Face* faces = &view.left;
    for(int i = 0; i < 6; i++) {
        faces[i].stride = imgView.stride();
        faces[i].data = &imgView(faceCoords[i].x, faceCoords[i].y);
    }
    return view;
}

template <int NC, typename T>
ImgView<NC, T> CubeImgView<NC, T>::operator[](ECubeImgFace eFace) {
    const int i = (int)eFace;
    assert(i >= 0 && i < 6);
    Face& face = (&left)[i];
    return ImgView<NC, T>(sidePixels, sidePixels, face.stride, face.data);
}
template <int NC, typename T>
ImgView<NC, const T> CubeImgView<NC, T>::operator[](ECubeImgFace eFace)const {
    const int i = (int)eFace;
    assert(i >= 0 && i < 6);
    const Face& face = (&left)[i];
    return ImgView<NC, const T>(sidePixels, sidePixels, face.stride, face.data);
}


template <int NC, typename T>
Img<NC, T>::Img(Img&& o)
    : ImgView<NC, T>(o)
{
    o._data = nullptr;
}

template <int NC, typename T>
Img<NC, T>& Img<NC, T>::operator=(Img&& o)
{
    free(Img<NC, T>::_data);
    Img<NC, T>::_w = o._w;
    Img<NC, T>::_h = o._h;
    Img<NC, T>::_stride = o._stride;
    Img<NC, T>::_data = o._data;
    o._data = nullptr;
    return *this;
}

}
