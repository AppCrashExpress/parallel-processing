#include "polygon.h"

Polygon::Polygon() {
    _texture_id = -1;
}

Polygon::Polygon(const Vector3D& a,
                 const Vector3D& b,
                 const Vector3D& c) {
    _points[0] = a;
    _points[1] = b;
    _points[2] = c;
    _calc_new_norm();

    _texture_id = -1;
}

Polygon::Polygon(const Vector3D& a,
                 const Vector3D& b,
                 const Vector3D& c,
                 int texture_id,
                 const TexCoord& tex_a,
                 const TexCoord& tex_b,
                 const TexCoord& tex_c) {
    _points[0] = a;
    _points[1] = b;
    _points[2] = c;
    _calc_new_norm();

    _texture_id = texture_id;
    _tex_coords[0] = tex_a;
    _tex_coords[1] = tex_b;
    _tex_coords[2] = tex_c;
}

void Polygon::set_a(const Vector3D& a) {
    _points[0] = a;
    _calc_new_norm();
}
void Polygon::set_b(const Vector3D& b) {
    _points[1] = b;
    _calc_new_norm();
}
void Polygon::set_c(const Vector3D& c) {
    _points[2] = c;
    _calc_new_norm();
}

const Vector3D& Polygon::a() const {
    return _points[0];
}
const Vector3D& Polygon::b() const {
    return _points[1];
}
const Vector3D& Polygon::c() const {
    return _points[2];
}

const Vector3D& Polygon::norm() const {
    return _norm;
}

std::pair<Vector3D, bool> Polygon::find_intersection(const Ray& ray, float eps) const {
    Vector3D res_vec;

    Vector3D e1 = b() - a();
    Vector3D e2 = c() - a();

    Vector3D h = Vector3D::cross(ray.dir, e2);
    float div = Vector3D::dot(h, e1);

    if (div > -eps && div < eps) {
        return {res_vec, false};
    }

    float coef = 1.0f / div;
    Vector3D s = ray.pos - a();
    float u = coef * Vector3D::dot(s, h);
    if (u < 0.0 || u > 1.0) {
        return {res_vec, false};
    }
    res_vec[1] = u;

    Vector3D q = Vector3D::cross(s, e1);
    float v = coef * Vector3D::dot(q, ray.dir);
    if (v < 0.0 || u + v > 1.0) {
        return {res_vec, false};
    }
    res_vec[2] = v;

    float t = coef * Vector3D::dot(q, e2);

    if (t > eps) {
        res_vec[0] = t;
        return {res_vec, true};
    } else {
        return {res_vec, false};
    }
}

int Polygon::get_texture_id() const {
    return _texture_id;
}

TexCoord Polygon::interpolate_tex(float u, float v) const {
    TexCoord res;

    res.first = (1 - u - v) * _tex_coords[0].first +
        u * _tex_coords[1].first +
        v * _tex_coords[2].first;
    res.second = (1 - u - v) * _tex_coords[0].second +
        u * _tex_coords[1].second +
        v * _tex_coords[2].second;

    return res;
}

void Polygon::_calc_new_norm() {
    Vector3D u = b() - a();
    Vector3D v = c() - a();
    _norm = Vector3D::cross(u, v).get_norm();
}
