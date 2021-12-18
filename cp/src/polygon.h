#ifndef POLYGON_H
#define POLYGON_H

#include <utility>

#include "linear.h"
#include "ray.h"
#include "tgaimage.h"
#include "texture.h"

class Polygon {
public:
    Polygon();

    Polygon(const Vector3D& a,
            const Vector3D& b,
            const Vector3D& c);

    Polygon(const Vector3D& a,
            const Vector3D& b,
            const Vector3D& c,
            int texture_id,
            const TexCoord& tex_a,
            const TexCoord& tex_b,
            const TexCoord& tex_c);

    void set_a(const Vector3D& a);
    void set_b(const Vector3D& b);
    void set_c(const Vector3D& c);

    const Vector3D& a() const;
    const Vector3D& b() const;
    const Vector3D& c() const;

    int get_texture_id() const;

    const Vector3D& norm() const;

    std::pair<Vector3D, bool> find_intersection(const Ray& ray, float eps) const;
    TexCoord interpolate_tex(float u, float v) const;
private:
    void _calc_new_norm();

    Vector3D _points[3];
    Vector3D _norm;
    TexCoord _tex_coords[3];
    int _texture_id;
};

#endif
