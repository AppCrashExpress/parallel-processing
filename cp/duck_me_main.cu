#include <vector>
#include <math.h>
#include <iostream>
#include <fstream>

#include <memory>

#pragma pack(push,1)
struct TGA_Header {
    std::uint8_t  idlength{};
    std::uint8_t  colormaptype{};
    std::uint8_t  datatypecode{};
    std::uint16_t colormaporigin{};
    std::uint16_t colormaplength{};
    std::uint8_t  colormapdepth{};
    std::uint16_t x_origin{};
    std::uint16_t y_origin{};
    std::uint16_t width{};
    std::uint16_t height{};
    std::uint8_t  bitsperpixel{};
    std::uint8_t  imagedescriptor{};
};
#pragma pack(pop)

struct TGAColor {
    std::uint8_t bgra[4] = {0,0,0,0};
    std::uint8_t bytespp = {0};

    TGAColor() = default;
    TGAColor(const std::uint8_t R, const std::uint8_t G, const std::uint8_t B, const std::uint8_t A=255) : bgra{B,G,R,A}, bytespp(4) { }
    TGAColor(const std::uint8_t v) : bgra{v,0,0,0}, bytespp(1) { }

    TGAColor(const std::uint8_t *p, const std::uint8_t bpp) : bgra{0,0,0,0}, bytespp(bpp) {
        for (int i=0; i<bpp; i++)
            bgra[i] = p[i];
    }

    std::uint8_t& operator[](const int i) { return bgra[i]; }

    TGAColor operator *(const double intensity) const {
        TGAColor res = *this;
        double clamped = std::max(0., std::min(intensity, 1.));
        for (int i=0; i<4; i++) res.bgra[i] = bgra[i]*clamped;
        return res;
    }
};

class TGAImage {
protected:
    std::vector<std::uint8_t> data;
    int width;
    int height;
    int bytespp;

    bool   load_rle_data(std::ifstream &in);
    bool unload_rle_data(std::ofstream &out) const;
public:
    enum Format { GRAYSCALE=1, RGB=3, RGBA=4 };

    TGAImage();
    TGAImage(const int w, const int h, const int bpp);
    bool read_tga_file(const std::string filename);
    bool write_tga_file(const std::string filename, const bool vflip=true, const bool rle=true) const;
    void flip_horizontally();
    void flip_vertically();
    void scale(const int w, const int h);
    TGAColor get(const int x, const int y) const;
    void set(const int x, const int y, const TGAColor &c);
    int get_width() const;
    int get_height() const;
    int get_bytespp();
    std::uint8_t *buffer();
    void clear();
};



class Texture;
class Ray;
class Vector3D;

struct TexCoord{
    float first; 
    float second;
    __host__ __device__
    TexCoord(){}
    __host__ __device__
    TexCoord(float l, float r): first(l), second(r) {}
};

class Vector3D {
public:
    __host__ __device__
    Vector3D() : _data{0,0,0} { }
    __host__ __device__
    Vector3D(float x, float y, float z) : _data{x,y,z} { }

    __host__ __device__
    float& operator[](int i); 
    __host__ __device__
    float  operator[](int i) const;

    __host__ __device__
    Vector3D& operator+=(const Vector3D& other);
    __host__ __device__
    Vector3D  operator+(const Vector3D& other) const;

    __host__ __device__
    Vector3D& operator-=(const Vector3D& other);
    __host__ __device__
    Vector3D  operator-(const Vector3D& other) const;

    __host__ __device__
    Vector3D& operator*=(const Vector3D& other);
    __host__ __device__
    Vector3D  operator*(const Vector3D& other) const;

    __host__ __device__
    Vector3D& operator/=(float val);
    __host__ __device__
    Vector3D  operator/(float val) const;

    __host__ __device__
    void set_x(float x);
    __host__ __device__
    void set_y(float y);
    __host__ __device__
    void set_z(float z);

    __host__ __device__
    float x() const;
    __host__ __device__
    float y() const;
    __host__ __device__
    float z() const;

    __host__ __device__
    float get_length() const;

    __host__ __device__
    Vector3D get_norm() const;

    __host__ __device__
    static float dot(const Vector3D& lhs, const Vector3D& rhs) {
        float res = 0;
        res += lhs.x() * rhs.x();
        res += lhs.y() * rhs.y();
        res += lhs.z() * rhs.z();
        return res;
    }

    __host__ __device__
    static Vector3D cross(const Vector3D& lhs, const Vector3D& rhs);

    __host__ __device__
    static Vector3D reflect(const Vector3D& dir, const Vector3D& norm);

private:
    float _data[3];
};

__host__ __device__
Vector3D operator*(float factor, const Vector3D& vec);
__host__ __device__
Vector3D operator*(const Vector3D& vec, float factor);

class Matrix4x4 {
public:
    __host__ __device__
    Matrix4x4() : _data{} {
        set_identity();
    }

    __host__ __device__
    float& operator()(int row, int col);
    __host__ __device__
    float operator()(int row, int col) const;

    __host__ __device__
    void set_identity();
    __host__ __device__
    void translate(const Vector3D& pos);
    __host__ __device__
    void scale(const Vector3D& pos);
    __host__ __device__
    void look_at(const Vector3D& pos,
                 const Vector3D& target,
                 const Vector3D& up_dir);

    __host__ __device__
    Matrix4x4  operator* (const Matrix4x4& other) const;
    __host__ __device__
    Matrix4x4& operator*=(const Matrix4x4& other);

private:
    float _data[4][4];
};
struct pair_Vector3D_bool{
    Vector3D first;
    bool second;
};


class Polygon {
public:
    __host__ __device__
    Polygon();

    __host__ __device__
    Polygon(const Vector3D& a,
            const Vector3D& b,
            const Vector3D& c);

    __host__ __device__
    Polygon(const Vector3D& a,
            const Vector3D& b,
            const Vector3D& c,
            int texture_id,
            const TexCoord& tex_a,
            const TexCoord& tex_b,
            const TexCoord& tex_c);

    __host__ __device__
    void set_a(const Vector3D& a);
    __host__ __device__
    void set_b(const Vector3D& b);
    __host__ __device__
    void set_c(const Vector3D& c);

    __host__ __device__
    const Vector3D& a() const;
    __host__ __device__
    const Vector3D& b() const;
    __host__ __device__
    const Vector3D& c() const;

    __host__ __device__
    int get_texture_id() const;

    __host__ __device__
    const Vector3D& norm() const;

    __host__ __device__
    pair_Vector3D_bool find_intersection(const Ray& ray, float eps) const;
    __host__ __device__
    TexCoord interpolate_tex(float u, float v) const;
private:
    __host__ __device__
    void _calc_new_norm();

    Vector3D _points[3];
    Vector3D _norm;
    TexCoord _tex_coords[3];
    int _texture_id;
};

class Ray {
public:

    __host__ __device__
    Ray(): use(true), consid_coef(1.0){}
    __host__ __device__
    Ray(const Vector3D& p, const Vector3D& d, int _x, int _y): pos(p), dir(d), x(_x), y(_y), use(true), consid_coef(1.0){}



    Vector3D pos;
    Vector3D dir;
    Vector3D color;
    bool use;
    float consid_coef;
    int x;
    int y;


private:

};





struct Pixel {
    __host__ __device__
    Pixel() {
        r = 0;
        g = 0;
        b = 0;
        reflect = 0.0f;
        refract = 0.0f;
    }

    __host__ __device__
    Pixel(unsigned char r,
          unsigned char g,
          unsigned char b,
          float reflect,
          float refract) {
        this->r = r;
        this->g = g;
        this->b = b;
        this->reflect = reflect;
        this->refract = refract;
    }

    unsigned char r;
    unsigned char g;
    unsigned char b;
    float reflect;
    float refract;
};

class Texture {
public:
    __host__ __device__
    Texture(){}
    __host__ __device__
    Texture(int w, int h, Pixel *data) {
        _w = w;
        _h = h;
        //_map = new Pixel[w * h];
        for (int i = 0 ; i < w*h; ++i ){
            _map[i] = data[i];
        }
    }

    // __host__ __device__
    // ~Texture(){
    //     delete [] _map;
    // }

    __host__ __device__
    Pixel at(TexCoord coord) const {
        int p_x = _w * coord.first;
        int p_y = _h * coord.second;

        return _map[p_y * _w + p_x];
    }

private:
    Pixel _map[256];
    int _w;
    int _h;
};

class Light {
public:
    Vector3D pos;
    Vector3D color;
};


__host__ __device__
float& Vector3D::operator[](int i) {
    return _data[i];
}
__host__ __device__
float Vector3D::operator[](int i) const {
    return _data[i];
}

__host__ __device__
Vector3D& Vector3D::operator+=(const Vector3D& other) {
    for (int i = 0; i < 3; ++i) {
        _data[i] += other._data[i];
    }
    return *this;
}
__host__ __device__
Vector3D Vector3D::operator+(const Vector3D& other) const {
    Vector3D res = *this;
    res += other;
    return res;
}

__host__ __device__
Vector3D& Vector3D::operator-=(const Vector3D& other) {
    for (int i = 0; i < 3; ++i) {
        _data[i] -= other._data[i];
    }
    return *this;
}
__host__ __device__
Vector3D Vector3D::operator-(const Vector3D& other) const {
    Vector3D res = *this;
    res -= other;
    return res;
}

__host__ __device__
Vector3D& Vector3D::operator*=(const Vector3D& other) {
    for (int i = 0; i < 3; ++i) {
        _data[i] *= other._data[i];
    }
    return *this;
}
__host__ __device__
Vector3D Vector3D::operator*(const Vector3D& other) const {
    Vector3D res = *this;
    res *= other;
    return res;
}

__host__ __device__
Vector3D& Vector3D::operator/=(float val) {
    for (int i = 0; i < 3; ++i) {
        _data[i] /= val;
    }
    return *this;
}
__host__ __device__
Vector3D Vector3D::operator/(float val) const {
    Vector3D res = *this;
    res /= val;
    return res;
}

__host__ __device__
void Vector3D::set_x(float x) {
    _data[0] = x;
}
__host__ __device__
void Vector3D::set_y(float y) {
    _data[1] = y;
}
__host__ __device__
void Vector3D::set_z(float z) {
    _data[2] = z;
}

__host__ __device__
float Vector3D::x() const {
    return _data[0];
}

__host__ __device__
float Vector3D::y() const {
    return _data[1];
}

__host__ __device__
float Vector3D::z() const {
    return _data[2];
}

__host__ __device__
float Vector3D::get_length() const {
    return sqrt(dot(*this, *this));
}

__host__ __device__
Vector3D Vector3D::get_norm() const {
    float len = this->get_length();
    return *this / len;
}

__host__ __device__
Vector3D Vector3D::cross(const Vector3D& lhs, const Vector3D& rhs) {
    Vector3D res;
    //  i  j  k
    // ax ay az
    // bx by bz
    res.set_x(lhs.y() * rhs.z() - lhs.z() * rhs.y());
    res.set_y(lhs.z() * rhs.x() - lhs.x() * rhs.z());
    res.set_z(lhs.x() * rhs.y() - lhs.y() * rhs.x());

    return res;
}

__host__ __device__
Vector3D Vector3D::reflect(const Vector3D& dir, const Vector3D& norm) {
    // Math stackexchange 13261
    float dot_prod = Vector3D::dot(dir, norm);
    return dir - 2 * dot_prod * norm;
}

__host__ __device__
Vector3D operator*(float factor, const Vector3D& vec) {
    Vector3D res;
    res.set_x(vec.x() * factor);
    res.set_y(vec.y() * factor);
    res.set_z(vec.z() * factor);
    return res;
}

__host__ __device__
Vector3D operator*(const Vector3D& vec, float factor) {
    return (factor * vec);
}


__host__ __device__
float& Matrix4x4::operator()(int row, int col) {
    return _data[row][col];
}
__host__ __device__
float Matrix4x4::operator()(int row, int col) const {
    return _data[row][col];
}

__host__ __device__
void Matrix4x4::set_identity() {
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            _data[i][j] = 0.0f;
        }
        _data[i][i] = 1.0f;
    }
}

__host__ __device__
void Matrix4x4::translate(const Vector3D& pos) {
    Matrix4x4 other;

    other._data[0][3] = pos.x();
    other._data[1][3] = pos.y();
    other._data[2][3] = pos.z();

    operator*=(other);
}

__host__ __device__
void Matrix4x4::scale(const Vector3D& pos) {
    Matrix4x4 other;

    other._data[0][0] = pos.x();
    other._data[1][1] = pos.y();
    other._data[2][2] = pos.z();

    operator*=(other);
}

__host__ __device__
void Matrix4x4::look_at(
        const Vector3D& eye,
        const Vector3D& center,
        const Vector3D& up) {
    Matrix4x4 other;

    Vector3D z = (eye - center).get_norm();
    Vector3D x = Vector3D::cross(up, z).get_norm();
    Vector3D y = Vector3D::cross(z, x);

    other._data[0][0] = x.x();
    other._data[0][1] = x.y();
    other._data[0][2] = x.z();

    other._data[1][0] = y.x();
    other._data[1][1] = y.y();
    other._data[1][2] = y.z();

    other._data[2][0] = z.x();
    other._data[2][1] = z.y();
    other._data[2][2] = z.z();

    other._data[3][3] = 1.0f;

    operator*=(other);
}

__host__ __device__
Matrix4x4 Matrix4x4::operator*(const Matrix4x4& other) const {
    Matrix4x4 res;

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            res(i, j) = 0;
            for (int k = 0; k < 4; ++k) {
                res(i, j) += (*this)(i, k) * other(k, j);
            }
        }
    }

    return res;
}


__host__ __device__
Matrix4x4& Matrix4x4::operator*=(const Matrix4x4& other) {
    *this = (*this) * other;
    return *this;
}

__host__ __device__
Vector3D operator*(const Vector3D& vec, const Matrix4x4& matr) {
    float vec_padded[4];
    for (int i = 0; i < 3; ++i) {
        vec_padded[i] = vec[i];
    }
    vec_padded[3] = 1.0f;

    Vector3D res;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            res[i] += vec_padded[j] * matr(j, i);
        }
    }

    return res;
}

__host__ __device__
Vector3D operator*(const Matrix4x4& matr, const Vector3D& vec) {
    float vec_padded[4];
    for (int i = 0; i < 3; ++i) {
        vec_padded[i] = vec[i];
    }
    vec_padded[3] = 1.0f;

    Vector3D res;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            res[i] += matr(i, j) * vec_padded[j];
        }
    }

    return res;
}





__host__ __device__
Vector3D operator*(const Matrix4x4& matr, const Vector3D& vec);
__host__ __device__
Vector3D operator*(const Vector3D& vec, const Matrix4x4& matr);

std::vector<Polygon> construct_cube(const Vector3D& center, float outer_radius, int texture_id) {
    std::vector<Polygon> polys(12);

    const Vector3D v[8] = {
        Vector3D( 1,  1,  1),
        Vector3D(-1,  1,  1),
        Vector3D(-1,  1, -1),
        Vector3D( 1,  1, -1),
        Vector3D( 1, -1,  1),
        Vector3D(-1, -1,  1),
        Vector3D(-1, -1, -1),
        Vector3D( 1, -1, -1),
    };

    const TexCoord tc[4] = {
        TexCoord(0, 0),
        TexCoord(0, 1),
        TexCoord(1, 0),
        TexCoord(1, 1),
    };

    // Top face
    polys[0] = Polygon(v[0], v[3], v[1], texture_id, tc[0], tc[1], tc[2]);
    polys[1] = Polygon(v[2], v[1], v[3], texture_id, tc[3], tc[2], tc[1]);

    // Bottom face
    polys[2] = Polygon(v[4], v[5], v[7], texture_id, tc[0], tc[1], tc[2]);
    polys[3] = Polygon(v[6], v[7], v[5], texture_id, tc[3], tc[2], tc[1]);

    // Front face
    polys[4] = Polygon(v[0], v[1], v[4], texture_id, tc[0], tc[1], tc[2]);
    polys[5] = Polygon(v[5], v[4], v[1], texture_id, tc[3], tc[2], tc[1]);

    // Back face
    polys[6] = Polygon(v[3], v[7], v[2], texture_id, tc[0], tc[1], tc[2]);
    polys[7] = Polygon(v[6], v[2], v[7], texture_id, tc[3], tc[2], tc[1]);

    // Left face
    polys[8] = Polygon(v[0], v[4], v[3], texture_id, tc[0], tc[1], tc[2]);
    polys[9] = Polygon(v[7], v[3], v[4], texture_id, tc[3], tc[2], tc[1]);

    // Right face
    polys[10] = Polygon(v[1], v[2], v[5], texture_id, tc[0], tc[1], tc[2]);
    polys[11] = Polygon(v[6], v[5], v[2], texture_id, tc[3], tc[2], tc[1]);

    Matrix4x4 world_model;
    world_model.translate(center);
    world_model.scale(Vector3D(outer_radius, outer_radius, outer_radius));
    for (int i = 0; i < 12; ++i) {
        Polygon& poly = polys[i];
        poly.set_a(world_model * poly.a());
        poly.set_b(world_model * poly.b());
        poly.set_c(world_model * poly.c());
    }

    return polys;
}


std::vector<Polygon> construct_floor(const Vector3D& center,
                                     float length,
                                     int texture_id) {
    std::vector<Polygon> floor(2);

    const Vector3D v[8] = {
        Vector3D( 1,  0,  1),
        Vector3D(-1,  0,  1),
        Vector3D(-1,  0, -1),
        Vector3D( 1,  0, -1),
    };

    const TexCoord tc[4] = {
        TexCoord(0, 0),
        TexCoord(0, 1),
        TexCoord(1, 0),
        TexCoord(1, 1),
    };

    floor[0] = Polygon(v[0], v[3], v[1], texture_id, tc[0], tc[1], tc[2]);
    floor[1] = Polygon(v[2], v[1], v[3], texture_id, tc[3], tc[2], tc[1]);

    Matrix4x4 world_model;
    world_model.translate(center);
    world_model.scale(Vector3D(length, length, length));
    for (int i = 0; i < 2; ++i) {
        Polygon& poly = floor[i];
        poly.set_a(world_model * poly.a());
        poly.set_b(world_model * poly.b());
        poly.set_c(world_model * poly.c());
    }
    
    return floor;
}


std::vector<Polygon> construct_cube(const Vector3D& center,
                                    float outer_radius,
                                    int texture_id);

std::vector<Polygon> construct_floor(const Vector3D& center,
                                     float length,
                                     int texture_id);



__host__ __device__
Polygon::Polygon() {
    _texture_id = -1;
}

__host__ __device__
Polygon::Polygon(const Vector3D& a,
                 const Vector3D& b,
                 const Vector3D& c) {
    _points[0] = a;
    _points[1] = b;
    _points[2] = c;
    _calc_new_norm();

    _texture_id = -1;
}

__host__ __device__
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

__host__ __device__
void Polygon::set_a(const Vector3D& a) {
    _points[0] = a;
    _calc_new_norm();
}
__host__ __device__
void Polygon::set_b(const Vector3D& b) {
    _points[1] = b;
    _calc_new_norm();
}
__host__ __device__
void Polygon::set_c(const Vector3D& c) {
    _points[2] = c;
    _calc_new_norm();
}

__host__ __device__
const Vector3D& Polygon::a() const {
    return _points[0];
}
__host__ __device__
const Vector3D& Polygon::b() const {
    return _points[1];
}
__host__ __device__
const Vector3D& Polygon::c() const {
    return _points[2];
}

__host__ __device__
const Vector3D& Polygon::norm() const {
    return _norm;
}

__host__ __device__
pair_Vector3D_bool Polygon::find_intersection(const Ray& ray, float eps) const {
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

__host__ __device__
int Polygon::get_texture_id() const {
    return _texture_id;
}

__host__ __device__
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

__host__ __device__
void Polygon::_calc_new_norm() {
    Vector3D u = b() - a();
    Vector3D v = c() - a();
    _norm = Vector3D::cross(u, v).get_norm();
}




TGAImage::TGAImage() : data(), width(0), height(0), bytespp(0) {}
TGAImage::TGAImage(const int w, const int h, const int bpp) : data(w*h*bpp, 0), width(w), height(h), bytespp(bpp) {}

bool TGAImage::read_tga_file(const std::string filename) {
    std::ifstream in;
    in.open (filename, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "can't open file " << filename << "\n";
        in.close();
        return false;
    }
    TGA_Header header;
    in.read(reinterpret_cast<char *>(&header), sizeof(header));
    if (!in.good()) {
        in.close();
        std::cerr << "an error occured while reading the header\n";
        return false;
    }
    width   = header.width;
    height  = header.height;
    bytespp = header.bitsperpixel>>3;
    if (width<=0 || height<=0 || (bytespp!=GRAYSCALE && bytespp!=RGB && bytespp!=RGBA)) {
        in.close();
        std::cerr << "bad bpp (or width/height) value\n";
        return false;
    }
    size_t nbytes = bytespp*width*height;
    data = std::vector<std::uint8_t>(nbytes, 0);
    if (3==header.datatypecode || 2==header.datatypecode) {
        in.read(reinterpret_cast<char *>(data.data()), nbytes);
        if (!in.good()) {
            in.close();
            std::cerr << "an error occured while reading the data\n";
            return false;
        }
    } else if (10==header.datatypecode||11==header.datatypecode) {
        if (!load_rle_data(in)) {
            in.close();
            std::cerr << "an error occured while reading the data\n";
            return false;
        }
    } else {
        in.close();
        std::cerr << "unknown file format " << (int)header.datatypecode << "\n";
        return false;
    }
    if (!(header.imagedescriptor & 0x20))
        flip_vertically();
    if (header.imagedescriptor & 0x10)
        flip_horizontally();
    std::cerr << width << "x" << height << "/" << bytespp*8 << "\n";
    in.close();
    return true;
}

bool TGAImage::load_rle_data(std::ifstream &in) {
    size_t pixelcount = width*height;
    size_t currentpixel = 0;
    size_t currentbyte  = 0;
    TGAColor colorbuffer;
    do {
        std::uint8_t chunkheader = 0;
        chunkheader = in.get();
        if (!in.good()) {
            std::cerr << "an error occured while reading the data\n";
            return false;
        }
        if (chunkheader<128) {
            chunkheader++;
            for (int i=0; i<chunkheader; i++) {
                in.read(reinterpret_cast<char *>(colorbuffer.bgra), bytespp);
                if (!in.good()) {
                    std::cerr << "an error occured while reading the header\n";
                    return false;
                }
                for (int t=0; t<bytespp; t++)
                    data[currentbyte++] = colorbuffer.bgra[t];
                currentpixel++;
                if (currentpixel>pixelcount) {
                    std::cerr << "Too many pixels read\n";
                    return false;
                }
            }
        } else {
            chunkheader -= 127;
            in.read(reinterpret_cast<char *>(colorbuffer.bgra), bytespp);
            if (!in.good()) {
                std::cerr << "an error occured while reading the header\n";
                return false;
            }
            for (int i=0; i<chunkheader; i++) {
                for (int t=0; t<bytespp; t++)
                    data[currentbyte++] = colorbuffer.bgra[t];
                currentpixel++;
                if (currentpixel>pixelcount) {
                    std::cerr << "Too many pixels read\n";
                    return false;
                }
            }
        }
    } while (currentpixel < pixelcount);
    return true;
}

bool TGAImage::write_tga_file(const std::string filename, const bool vflip, const bool rle) const {
    std::uint8_t developer_area_ref[4] = {0, 0, 0, 0};
    std::uint8_t extension_area_ref[4] = {0, 0, 0, 0};
    std::uint8_t footer[18] = {'T','R','U','E','V','I','S','I','O','N','-','X','F','I','L','E','.','\0'};
    std::ofstream out;
    out.open (filename, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "can't open file " << filename << "\n";
        out.close();
        return false;
    }
    TGA_Header header;
    header.bitsperpixel = bytespp<<3;
    header.width  = width;
    header.height = height;
    header.datatypecode = (bytespp==GRAYSCALE?(rle?11:3):(rle?10:2));
    header.imagedescriptor = vflip ? 0x00 : 0x20; // top-left or bottom-left origin
    out.write(reinterpret_cast<const char *>(&header), sizeof(header));
    if (!out.good()) {
        out.close();
        std::cerr << "can't dump the tga file\n";
        return false;
    }
    if (!rle) {
        out.write(reinterpret_cast<const char *>(data.data()), width*height*bytespp);
        if (!out.good()) {
            std::cerr << "can't unload raw data\n";
            out.close();
            return false;
        }
    } else {
        if (!unload_rle_data(out)) {
            out.close();
            std::cerr << "can't unload rle data\n";
            return false;
        }
    }
    out.write(reinterpret_cast<const char *>(developer_area_ref), sizeof(developer_area_ref));
    if (!out.good()) {
        std::cerr << "can't dump the tga file\n";
        out.close();
        return false;
    }
    out.write(reinterpret_cast<const char *>(extension_area_ref), sizeof(extension_area_ref));
    if (!out.good()) {
        std::cerr << "can't dump the tga file\n";
        out.close();
        return false;
    }
    out.write(reinterpret_cast<const char *>(footer), sizeof(footer));
    if (!out.good()) {
        std::cerr << "can't dump the tga file\n";
        out.close();
        return false;
    }
    out.close();
    return true;
}

// TODO: it is not necessary to break a raw chunk for two equal pixels (for the matter of the resulting size)
bool TGAImage::unload_rle_data(std::ofstream &out) const {
    const std::uint8_t max_chunk_length = 128;
    size_t npixels = width*height;
    size_t curpix = 0;
    while (curpix<npixels) {
        size_t chunkstart = curpix*bytespp;
        size_t curbyte = curpix*bytespp;
        std::uint8_t run_length = 1;
        bool raw = true;
        while (curpix+run_length<npixels && run_length<max_chunk_length) {
            bool succ_eq = true;
            for (int t=0; succ_eq && t<bytespp; t++)
                succ_eq = (data[curbyte+t]==data[curbyte+t+bytespp]);
            curbyte += bytespp;
            if (1==run_length)
                raw = !succ_eq;
            if (raw && succ_eq) {
                run_length--;
                break;
            }
            if (!raw && !succ_eq)
                break;
            run_length++;
        }
        curpix += run_length;
        out.put(raw?run_length-1:run_length+127);
        if (!out.good()) {
            std::cerr << "can't dump the tga file\n";
            return false;
        }
        out.write(reinterpret_cast<const char *>(data.data()+chunkstart), (raw?run_length*bytespp:bytespp));
        if (!out.good()) {
            std::cerr << "can't dump the tga file\n";
            return false;
        }
    }
    return true;
}

TGAColor TGAImage::get(const int x, const int y) const {
    if (!data.size() || x<0 || y<0 || x>=width || y>=height)
        return {};
    return TGAColor(data.data()+(x+y*width)*bytespp, bytespp);
}

void TGAImage::set(int x, int y, const TGAColor &c) {
    if (!data.size() || x<0 || y<0 || x>=width || y>=height) return;
    memcpy(data.data()+(x+y*width)*bytespp, c.bgra, bytespp);
}

int TGAImage::get_bytespp() {
    return bytespp;
}

int TGAImage::get_width() const {
    return width;
}

int TGAImage::get_height() const {
    return height;
}

void TGAImage::flip_horizontally() {
    if (!data.size()) return;
    int half = width>>1;
    for (int i=0; i<half; i++) {
        for (int j=0; j<height; j++) {
            TGAColor c1 = get(i, j);
            TGAColor c2 = get(width-1-i, j);
            set(i, j, c2);
            set(width-1-i, j, c1);
        }
    }
}

void TGAImage::flip_vertically() {
    if (!data.size()) return;
    size_t bytes_per_line = width*bytespp;
    std::vector<std::uint8_t> line(bytes_per_line, 0);
    int half = height>>1;
    for (int j=0; j<half; j++) {
        size_t l1 = j*bytes_per_line;
        size_t l2 = (height-1-j)*bytes_per_line;
        std::copy(data.begin()+l1, data.begin()+l1+bytes_per_line, line.begin());
        std::copy(data.begin()+l2, data.begin()+l2+bytes_per_line, data.begin()+l1);
        std::copy(line.begin(), line.end(), data.begin()+l2);
    }
}

std::uint8_t *TGAImage::buffer() {
    return data.data();
}

void TGAImage::clear() {
    data = std::vector<std::uint8_t>(width*height*bytespp, 0);
}

void TGAImage::scale(int w, int h) {
    if (w<=0 || h<=0 || !data.size()) return;
    std::vector<std::uint8_t> tdata(w*h*bytespp, 0);
    int nscanline = 0;
    int oscanline = 0;
    int erry = 0;
    size_t nlinebytes = w*bytespp;
    size_t olinebytes = width*bytespp;
    for (int j=0; j<height; j++) {
        int errx = width-w;
        int nx   = -bytespp;
        int ox   = -bytespp;
        for (int i=0; i<width; i++) {
            ox += bytespp;
            errx += w;
            while (errx>=(int)width) {
                errx -= width;
                nx += bytespp;
                memcpy(tdata.data()+nscanline+nx, data.data()+oscanline+ox, bytespp);
            }
        }
        erry += h;
        oscanline += olinebytes;
        while (erry>=(int)height) {
            if (erry>=(int)height<<1) // it means we jump over a scanline
                memcpy(tdata.data()+nscanline+nlinebytes, tdata.data()+nscanline, nlinebytes);
            erry -= height;
            nscanline += nlinebytes;
        }
    }
    data = tdata;
    width = w;
    height = h;
}







struct IntersectType {
    Vector3D first; 
    int second;
    __host__ __device__
    IntersectType (){}
    __host__ __device__
    IntersectType (const Vector3D& l, const int r): first(l), second(r) {}
};
//


#define CSC(call)                           \
do {                                \
  cudaError_t res = call;                     \
  if (res != cudaSuccess) {                   \
    fprintf(stderr, "ERROR in %s:%d. Message: %s\n",      \
        __FILE__, __LINE__, cudaGetErrorString(res));   \
    exit(0);                          \
  }                               \
} while(0);



const size_t polygons_size = 50;  
const size_t lights_size   = 2; 
const size_t textures_size = 5; 

// __constant__  Polygon dev_polygons[polygons_size];
// __constant__ Light   dev_lights  [lights_size];
// __constant__ Texture dev_textures[textures_size];



Polygon polygons[polygons_size];
Light   lights[lights_size];
Texture textures[textures_size];

__device__ Polygon* dev_polygons;
__device__ Light*   dev_lights  ;
__device__ Texture* dev_textures;

void build_space() {
    Pixel pixels[4] = {
        {255,   0,   0, 0.0f, 0.7f,},
        {  0, 255,   0, 0.0f, 0.7f,},
        {  0,   0, 255, 0.0f, 0.7f,},
        {255, 255, 255, 0.0f, 0.7f,},
    };
    textures[0] = Texture(2, 2, pixels);

    Pixel pixel = {255, 0, 0, 0.0f, 0.0f};
    textures[1] = Texture(1, 1, &pixel);

    pixel = {0, 0, 255, 0.0f, 0.0f};
    textures[2] = Texture(1, 1, &pixel);

    pixel = {0, 255, 0, 0.0f, 0.0f};
    textures[3] = Texture(1, 1, &pixel);

    pixel = {100, 100, 100, 0.7f, 0.0f};
    textures[4] = Texture(1, 1, &pixel);

    std::vector<Polygon> cube; // 12 polys * 4

    cube = construct_cube(Vector3D(0, 0, 0), 1.0f, 0);
    for (int i = 0 ; i < 12 ; ++i) {
        polygons[i] = cube[i];
    }
    //polygons.insert(polygons.end(), cube.begin(), cube.end());


    cube = construct_cube(Vector3D(5.0f, 0, 0), 1.0f, 1);
    for (int i = 0 ; i < 12 ; ++i) {
        polygons[i+12] = cube[i];
    }

    //polygons.insert(polygons.end(), cube.begin(), cube.end());

    cube = construct_cube(Vector3D(0, 5.0f, 0), 1.0f, 2);
    for (int i = 0 ; i < 12 ; ++i) {
        polygons[i+24] = cube[i];
    }
    //polygons.insert(polygons.end(), cube.begin(), cube.end());

    cube = construct_cube(Vector3D(0, 0, 5.0f), 1.0f, 3);
    for (int i = 0 ; i < 12 ; ++i) {
        polygons[i+36] = cube[i];
    }
    //polygons.insert(polygons.end(), cube.begin(), cube.end());

    std::vector<Polygon> floor = construct_floor(Vector3D(0.0f, -2.0f, 0.0f), 10, 4); // 2 polys
    for (int i = 0 ; i < 2 ; ++i) {
        polygons[i+48] = floor[i];
    }
    //polygons.insert(polygons.end(), floor.begin(), floor.end());

    Vector3D light_color = Vector3D(1.0f, 1.0f, 1.0f);

    //lights[0] = {Vector3D(2.0f, 0.0f, 0.0f), light_color};
    lights[0] =  {Vector3D(0.0f, 2.0f, 0.0f), light_color};                
    light_color = Vector3D(0.0f, 0.0f, 0.0f);
    lights[1] =  {Vector3D(0.0f, 2.0f, 0.0f), light_color};                
}

void
set_on_gpu(){

    CSC(cudaMalloc(&dev_polygons, sizeof(Polygon)*polygons_size));
    CSC(cudaMalloc(&dev_lights,   sizeof(Light)  *lights_size));
    CSC(cudaMalloc(&dev_textures, sizeof(Texture)*textures_size));

    // CSC(cudaMemcpy(dev_polygons, polygons, sizeof(Polygon)*polygons_size,  cudaMemcpyHostToDevice));
    // CSC(cudaMemcpy(dev_lights,   lights,   sizeof(Light)*lights_size,      cudaMemcpyHostToDevice));
    // CSC(cudaMemcpy(dev_textures, textures, sizeof(Texture)*textures_size,  cudaMemcpyHostToDevice));

    Polygon* ukazatel;
    CSC(cudaMalloc(&ukazatel, sizeof(Polygon)*polygons_size));
    CSC(cudaMemcpy(ukazatel, polygons, sizeof(Polygon)*polygons_size,  cudaMemcpyHostToDevice));
    CSC(cudaMemcpyToSymbol(dev_polygons, &ukazatel, sizeof(Polygon*), 0, cudaMemcpyHostToDevice));
    
    Light* ukazatel2;
    CSC(cudaMalloc(&ukazatel2, sizeof(Light)*lights_size));
    CSC(cudaMemcpy(ukazatel2, lights, sizeof(Light)*lights_size,  cudaMemcpyHostToDevice));
    CSC(cudaMemcpyToSymbol(dev_lights, &ukazatel2, sizeof(Light*), 0, cudaMemcpyHostToDevice));
    

    Texture* ukazatel3;
    CSC(cudaMalloc(&ukazatel3, sizeof(Texture)*textures_size));
    CSC(cudaMemcpy(ukazatel3, textures, sizeof(Texture)*textures_size,  cudaMemcpyHostToDevice));
 
    CSC(cudaMemcpyToSymbol(dev_textures, &ukazatel3, sizeof(Texture*), 0, cudaMemcpyHostToDevice));
}

void
free_from_gpu(){
    // CSC(cudaFree(dev_polygons));
    // CSC(cudaFree(dev_lights));
    // CSC(cudaFree(dev_textures));
}

__device__
IntersectType intersect_polygons(const Ray& ray) {
    Vector3D max_res;
    int max_poly = -1;

    max_res[0] = 1e10;
    for (int i = 0; i < polygons_size; ++i) {
        pair_Vector3D_bool res = dev_polygons[i].find_intersection(ray, 1e-7);
        if (res.second && max_res[0] > res.first[0]) {
            max_res = res.first;
            max_poly = i;
        }
    }

    return {max_res, max_poly};
}

__device__
Pixel retrieve_intersection_pixel(const IntersectType& intersection) {

    const float t = intersection.first[0];
    const float u = intersection.first[1];
    const float v = intersection.first[2];
    const int polygon_id = intersection.second;
    Vector3D color;

    const Texture& tex = dev_textures[dev_polygons[polygon_id].get_texture_id()];
    TexCoord tex_coord = dev_polygons[polygon_id].interpolate_tex(u, v);
    Pixel pix = tex.at(tex_coord);

    return pix;
}

__device__
void check_visible_lights(const Vector3D& pos, char* visible) {
    //std::vector<char> visible(lights_size, false);

    for (size_t i = 0; i < lights_size; ++i) {
        
        Vector3D dir = pos - dev_lights[i].pos;
        float t_max = dir.get_length() - 1e-4f;
        dir = dir.get_norm();

        Ray ray;
        ray.pos = dev_lights[i].pos;
        ray.dir = dir;
        IntersectType intersect = intersect_polygons(ray);

        float t = intersect.first[0];
        if (t < 0 || t >= t_max) {
        //    Pixel hit_pix = retrieve_intersection_pixel(intersect);
            visible[i] = true;
        }
    }

    //return visible;
}

__device__
Vector3D shade_pixel(const Pixel& pix,
                     const Vector3D& norm,
                     const Vector3D& intersect_pos) {
    Vector3D colors;

    char visible[lights_size]; 
    
    check_visible_lights(intersect_pos, visible);
    for (int l = 0; l < lights_size; ++l) {
        if (!visible[l]) {
            continue;
        }

        Vector3D light_dir = (dev_lights[l].pos - intersect_pos).get_norm();
        float strength = max(Vector3D::dot(norm, light_dir), 0.0f);

        if (strength > 1e-5) {
            Vector3D temp;
            temp.set_x(pix.r);
            temp.set_y(pix.g);
            temp.set_z(pix.b);
            colors += strength * dev_lights[l].color * temp;
        }
    }

    return colors;
}

__device__
Vector3D dev_cast_ray(Ray& ray, Ray& reflection_ray) {

    // const float coef = 1;
    IntersectType intersect = intersect_polygons(ray);
    const float t = intersect.first[0];
    const int polygon_id = intersect.second;
    Vector3D color;

    if (polygon_id != -1) {
        Vector3D intersect_pos = ray.pos + t * ray.dir;

        Pixel pix = retrieve_intersection_pixel(intersect);
        color += shade_pixel(pix, dev_polygons[polygon_id].norm(), intersect_pos);

        ray.color = color;
        // ray.consid_coef *=coef; 

        // Use small value just in case for now
        if (pix.reflect > 0.03f) {
            const Vector3D& norm = dev_polygons[polygon_id].norm();
            const Vector3D reflect_vec = Vector3D::reflect(ray.dir, norm);
            
            reflection_ray.pos = intersect_pos + 0.03 * norm;
            reflection_ray.dir = reflect_vec;

            reflection_ray.consid_coef =  pix.reflect * ray.consid_coef /** coef*/;
            reflection_ray.x = ray.x;
            reflection_ray.y = ray.y;

        }

    }else{
        //reflection_ray.use = false;
        ray.color = {0.0f, 0.0f, 0.0f};
        
    }

    return color;
}

__global__
void
kernel_do_rays(Ray* cur_rays, Ray* next_rays, size_t cur_size){
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    int offset = blockDim.x*gridDim.x;

    for(size_t i = idx ; i < cur_size; i+=offset) {
        //cur_rays[i].color = {255, 255, 255};
        dev_cast_ray(cur_rays[i], next_rays[i]);
    }
}


void
do_colors(Vector3D* image_buff, Ray* rays,const int rays_size, const int h){

    for (int i = 0; i < rays_size; ++i){
        int x = rays[i].x, 
            y = rays[i].y;

        image_buff[y*h+x] += rays[i].color*rays[i].consid_coef;
    }
}

int
compact(Ray* rays, int rays_size){
    int cnt=0; 
    for(int i = 0; i < rays_size; ++i)
        if(rays[i].use){
            rays[cnt++]=rays[i];
        }
    return cnt;
    
}

void 
set_image(Vector3D* image_buff, TGAImage& image, int w, int h ){
    for(int i = 0 ; i < w ; ++i)
        for(int j = 0 ; j < h ; ++j){
            const Vector3D& color_vec = image_buff[i*h+j];
            TGAColor color(
                std::min(color_vec.x(), 255.0f),
                std::min(color_vec.y(), 255.0f),
                std::min(color_vec.z(), 255.0f)
            );
            image.set(i, j, color);
        }


}
__host__
void render_gpu(const Vector3D& pc,
            const Vector3D& pv,
            int w, int h,
            float angle,
            TGAImage& image, 
            const int max_depth) {
    float dw = 2.0f / (w - 1);
    float dh = 2.0f / (h - 1);
    float z  = 1.0f / tan(angle * M_PI / 360.0f);
    
    Matrix4x4 look_matr;
    look_matr.look_at(pv, pc, Vector3D(0.0f, 1.0f, 0.0f));


    size_t pixel_cnt = w*h;
    
    std::unique_ptr<Vector3D[]> image_buff (new Vector3D[pixel_cnt]);
    std::unique_ptr<Ray[]>      cur_rays   (new Ray[pixel_cnt]);
    std::unique_ptr<Ray[]>      next_rays  (new Ray[pixel_cnt]);



    for (int i = 0; i < w; ++i) {
        for (int j = 0; j < h; ++j) {
            Vector3D v = Vector3D(-1.0f + dw * i, (-1.0f + dh * j) * h / w, z);
            Vector3D dir = (v * look_matr).get_norm();
            cur_rays[i*h+j] = Ray(pc, dir, j, i);
            // std::cerr<<dir.x()<<' '<<dir.y()<<' '<<dir.z()<<std::endl;
            // std::cerr<<pc.x()<<' '<<pc.y()<<' '<<pc.z()<<std::endl;

        }
    }

    Vector3D* dev_image_buff;
    CSC(cudaMalloc(&dev_image_buff, sizeof(Vector3D)*pixel_cnt));
    CSC(cudaMemcpy(dev_image_buff, image_buff.get(), sizeof(Vector3D)*pixel_cnt, cudaMemcpyHostToDevice));

    Ray* dev_cur_rays, *dev_next_rays;
    CSC(cudaMalloc(&dev_cur_rays,  sizeof(Ray)*pixel_cnt));
    CSC(cudaMalloc(&dev_next_rays, sizeof(Ray)*pixel_cnt));

    CSC(cudaMemcpy(dev_cur_rays, cur_rays.get(), sizeof(Ray)*pixel_cnt, cudaMemcpyHostToDevice));

    int cur_size = pixel_cnt;

    for (int depth = 0; depth < max_depth; ++depth) {
    
        kernel_do_rays<<<1024, 1024>>>(dev_cur_rays, dev_next_rays, cur_size);

        CSC(cudaMemcpy( cur_rays.get(), dev_cur_rays, sizeof(Ray)*pixel_cnt, cudaMemcpyDeviceToHost));

        do_colors(image_buff.get(), cur_rays.get(), cur_size, h);
        
        //cur_size = compact(next_rays.get(), cur_size);
        
        std::swap(dev_cur_rays, dev_next_rays);


    }
    set_image(image_buff.get(), image, w,h);
    
    CSC(cudaGetLastError());

    CSC(cudaFree(dev_image_buff));
    CSC(cudaFree(dev_cur_rays));
    CSC(cudaFree(dev_next_rays));
    
}

int main() {
    int w = 640;
    int h = 480;
    char buff[256];
    TGAImage image(w, h, TGAImage::Format::RGBA);

    build_space();
    set_on_gpu();

    int max_depth = 2;
    for (int k = 0; k < 10; ++k) {
        //max_depth++;
        Vector3D pc = Vector3D( 6.0f +3.0f*sin(k*0.1f),  3.0f + 3.0f*sin(k*0.1f), 3.0f + 3.0f*sin(k*0.2f));
        Vector3D pv = Vector3D(0.0f, 3.0f, 0.0f);
        render_gpu(pc, pv, w, h, 120.0, image, max_depth);

        sprintf(buff, "res/%d.tga", k);
        image.write_tga_file(buff, true, false);
        printf("%d: %s\n", k, buff);        
    }
    free_from_gpu();
}
