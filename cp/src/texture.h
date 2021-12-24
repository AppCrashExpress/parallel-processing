#ifndef TEXTURE_H
#define TEXTURE_H

#include <memory>
#include <cstring>

using TexCoord = std::pair<float, float>;

struct Pixel {
    Pixel() {
        r = 0;
        g = 0;
        b = 0;
        reflect = 0.0f;
        refract = 0.0f;
    }

    Pixel(float r, float g, float b,
          float reflect, float refract) {
        this->r = r;
        this->g = g;
        this->b = b;
        this->reflect = reflect;
        this->refract = refract;
    }

    float r;
    float g;
    float b;
    float reflect;
    float refract;
};

class Texture {
public:
    Texture(int w, int h, Pixel *data) {
        _w = w;
        _h = h;
        _map = std::unique_ptr<Pixel[]>(new Pixel[w * h]);
        std::memcpy(_map.get(), data, sizeof(Pixel) * w * h);
    }

    Pixel at(TexCoord coord) const {
        int p_x = _w * coord.first;
        int p_y = _h * coord.second;

        return _map[p_y * _w + p_x];
    }

private:
    std::unique_ptr<Pixel[]> _map;
    int _w;
    int _h;
};

#endif
