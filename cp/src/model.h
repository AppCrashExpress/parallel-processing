#ifndef MODEL_H
#define MODEL_H

#include <vector>

#include "linear.h"
#include "polygon.h"

std::vector<Polygon> construct_cube(const Vector3D& center,
                                    float outer_radius,
                                    int texture_id);

std::vector<Polygon> construct_floor(const Vector3D& center,
                                     float length,
                                     int texture_id);

#endif
