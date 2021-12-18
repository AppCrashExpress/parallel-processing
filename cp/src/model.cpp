#include "model.h"
#include "texture.h"

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
