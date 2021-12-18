#include "linear.h"

float& Vector3D::operator[](int i) {
    return _data[i];
}
float Vector3D::operator[](int i) const {
    return _data[i];
}

Vector3D& Vector3D::operator+=(const Vector3D& other) {
    for (int i = 0; i < 3; ++i) {
        _data[i] += other._data[i];
    }
    return *this;
}
Vector3D Vector3D::operator+(const Vector3D& other) const {
    Vector3D res = *this;
    res += other;
    return res;
}

Vector3D& Vector3D::operator-=(const Vector3D& other) {
    for (int i = 0; i < 3; ++i) {
        _data[i] -= other._data[i];
    }
    return *this;
}
Vector3D Vector3D::operator-(const Vector3D& other) const {
    Vector3D res = *this;
    res -= other;
    return res;
}

Vector3D& Vector3D::operator*=(const Vector3D& other) {
    for (int i = 0; i < 3; ++i) {
        _data[i] *= other._data[i];
    }
    return *this;
}
Vector3D Vector3D::operator*(const Vector3D& other) const {
    Vector3D res = *this;
    res *= other;
    return res;
}

Vector3D& Vector3D::operator/=(float val) {
    for (int i = 0; i < 3; ++i) {
        _data[i] /= val;
    }
    return *this;
}
Vector3D Vector3D::operator/(float val) const {
    Vector3D res = *this;
    res /= val;
    return res;
}

void Vector3D::set_x(float x) {
    _data[0] = x;
}
void Vector3D::set_y(float y) {
    _data[1] = y;
}
void Vector3D::set_z(float z) {
    _data[2] = z;
}

float Vector3D::x() const {
    return _data[0];
}

float Vector3D::y() const {
    return _data[1];
}

float Vector3D::z() const {
    return _data[2];
}

float Vector3D::get_length() const {
    return std::sqrt(dot(*this, *this));
}

Vector3D Vector3D::get_norm() const {
    float len = this->get_length();
    return *this / len;
}

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

Vector3D Vector3D::reflect(const Vector3D& dir, const Vector3D& norm) {
    // Math stackexchange 13261
    float dot_prod = Vector3D::dot(dir, norm);
    return dir - 2 * dot_prod * norm;
}

Vector3D operator*(float factor, const Vector3D& vec) {
    Vector3D res;
    res.set_x(vec.x() * factor);
    res.set_y(vec.y() * factor);
    res.set_z(vec.z() * factor);
    return res;
}

Vector3D operator*(const Vector3D& vec, float factor) {
    return (factor * vec);
}


float& Matrix4x4::operator()(int row, int col) {
    return _data[row][col];
}
float Matrix4x4::operator()(int row, int col) const {
    return _data[row][col];
}

void Matrix4x4::set_identity() {
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            _data[i][j] = 0.0f;
        }
        _data[i][i] = 1.0f;
    }
}

void Matrix4x4::translate(const Vector3D& pos) {
    Matrix4x4 other;

    other._data[0][3] = pos.x();
    other._data[1][3] = pos.y();
    other._data[2][3] = pos.z();

    operator*=(other);
}

void Matrix4x4::scale(const Vector3D& pos) {
    Matrix4x4 other;

    other._data[0][0] = pos.x();
    other._data[1][1] = pos.y();
    other._data[2][2] = pos.z();

    operator*=(other);
}

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


Matrix4x4& Matrix4x4::operator*=(const Matrix4x4& other) {
    *this = (*this) * other;
    return *this;
}

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
