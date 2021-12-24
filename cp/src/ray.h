#ifndef RAY_H
#define RAY_H

#include "linear.h"

class Ray {
public:
    Ray(): use(true), consid_coef(1.0){}
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

#endif
