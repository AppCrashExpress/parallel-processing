#ifndef LINEAR_H
#define LINEAR_H

#include <cmath>

class Vector3D {
public:
    Vector3D() : _data{} { }
    Vector3D(float x, float y, float z) : _data{x,y,z} { }

    float& operator[](int i); 
    float  operator[](int i) const;

    Vector3D& operator+=(const Vector3D& other);
    Vector3D  operator+(const Vector3D& other) const;

    Vector3D& operator-=(const Vector3D& other);
    Vector3D  operator-(const Vector3D& other) const;

    Vector3D& operator*=(const Vector3D& other);
    Vector3D  operator*(const Vector3D& other) const;

    Vector3D& operator/=(float val);
    Vector3D  operator/(float val) const;

    void set_x(float x);
    void set_y(float y);
    void set_z(float z);

    float x() const;
    float y() const;
    float z() const;

    float get_length() const;

    Vector3D get_norm() const;

    static float dot(const Vector3D& lhs, const Vector3D& rhs) {
        float res = 0;
        res += lhs.x() * rhs.x();
        res += lhs.y() * rhs.y();
        res += lhs.z() * rhs.z();
        return res;
    }

    static Vector3D cross(const Vector3D& lhs, const Vector3D& rhs);

    static Vector3D reflect(const Vector3D& dir, const Vector3D& norm);

private:
    float _data[3];
};

Vector3D operator*(float factor, const Vector3D& vec);
Vector3D operator*(const Vector3D& vec, float factor);

class Matrix4x4 {
public:
    Matrix4x4() : _data{} {
        set_identity();
    }

    float& operator()(int row, int col);
    float operator()(int row, int col) const;

    void set_identity();
    void translate(const Vector3D& pos);
    void scale(const Vector3D& pos);
    void look_at(const Vector3D& pos,
                 const Vector3D& target,
                 const Vector3D& up_dir);

    Matrix4x4  operator* (const Matrix4x4& other) const;
    Matrix4x4& operator*=(const Matrix4x4& other);

private:
    float _data[4][4];
};

Vector3D operator*(const Matrix4x4& matr, const Vector3D& vec);
Vector3D operator*(const Vector3D& vec, const Matrix4x4& matr);

#endif
