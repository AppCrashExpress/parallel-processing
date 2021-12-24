#include <iostream>
#include <vector>
#include <cmath>

#include "tgaimage.h"
#include "linear.h"
#include "polygon.h"
#include "ray.h"
#include "model.h"
#include "light.h"
#include "texture.h"

using IntersectType = std::pair<Vector3D, int>;

void build_space(std::vector<Polygon>& polys,
                 std::vector<Light>& lights,
                 std::vector<Texture>& textures) {
    Pixel pixels[4] = {
        {255,   0,   0, 0.0f, 0.7f,},
        {  0, 255,   0, 0.0f, 0.7f,},
        {  0,   0, 255, 0.0f, 0.7f,},
        {255, 255, 255, 0.0f, 0.7f,},
    };
    textures.push_back(Texture(2, 2, pixels));

    Pixel pixel;
    pixel.r = 255;
    pixel.b = 0;
    pixel.g = 0;
    pixel.reflect = 0.0f;
    pixel.refract = 0.0f;
    textures.push_back(Texture(1, 1, &pixel));

    pixel.r = 0;
    pixel.b = 255;
    pixel.g = 0;
    textures.push_back(Texture(1, 1, &pixel));

    pixel.r = 0;
    pixel.b = 0;
    pixel.g = 255;
    pixel.refract = 0.95;
    textures.push_back(Texture(1, 1, &pixel));

    pixel.r = 255;
    pixel.b = 255;
    pixel.g = 50;
    pixel.reflect = 0.25;
    pixel.refract = 0;
    textures.push_back(Texture(1, 1, &pixel));

    std::vector<Polygon> cube;

    cube = construct_cube(Vector3D(0, 0, 0), 1.0f, 0);
    polys.insert(polys.end(), cube.begin(), cube.end());

    cube = construct_cube(Vector3D(5.0f, 0, 0), 1.0f, 1);
    polys.insert(polys.end(), cube.begin(), cube.end());

    cube = construct_cube(Vector3D(0, 5.0f, 0), 1.0f, 2);
    polys.insert(polys.end(), cube.begin(), cube.end());

    cube = construct_cube(Vector3D(0, 0, 5.0f), 1.0f, 3);
    polys.insert(polys.end(), cube.begin(), cube.end());

    std::vector<Polygon> floor = construct_floor(Vector3D(0.0f, -2.0f, 0.0f), 10, 4);
    polys.insert(polys.end(), floor.begin(), floor.end());

    const Vector3D light_color = Vector3D(1.0f, 1.0f, 1.0f);

    lights = std::vector<Light>{{Vector3D(2.0f, 0.0f, 0.0f), light_color},
                                {Vector3D(0.0f, 2.0f, 0.0f), light_color}};
}

std::pair<Vector3D, int> intersect_polygons(const std::vector<Polygon>& polygons,
                            const Ray& ray) {
    Vector3D max_res;
    int max_poly = -1;

    max_res[0] = 1e10;
    for (int i = 0; i < polygons.size(); ++i) {
        std::pair<Vector3D, bool> res = polygons[i].find_intersection(ray, 1e-7);
        if (res.second && max_res[0] > res.first[0]) {
            max_res = res.first;
            max_poly = i;
        }
    }

    return {max_res, max_poly};
}

Pixel retrieve_intersection_pixel(const std::vector<Polygon>& polygons,
                                  const std::vector<Texture>& textures,
                                  const IntersectType& intersection) {

    const float t = intersection.first[0];
    const float u = intersection.first[1];
    const float v = intersection.first[2];
    const int polygon_id = intersection.second;
    Vector3D color;

    const Texture& tex = textures[ polygons[polygon_id].get_texture_id() ];
    TexCoord tex_coord = polygons[polygon_id].interpolate_tex(u, v);
    Pixel pix = tex.at(tex_coord);

    return pix;
}

std::vector<char> 
check_visible_lights(const std::vector<Polygon>& polygons,
        const std::vector<Texture>& textures,
        const std::vector<Light>& lights,
        const Vector3D& pos) {
    std::vector<char> visible(lights.size(), false);

    for (size_t i = 0; i < lights.size(); ++i) {
        Vector3D dir = pos - lights[i].pos;
        float t_max = dir.get_length() - 1e-4f;
        dir = dir.get_norm();

        Ray ray;
        ray.pos = lights[i].pos;
        ray.dir = dir;
        IntersectType intersect = intersect_polygons(polygons, ray);

        float t = intersect.first[0];
        if (t < 0 || t >= t_max) {
            Pixel hit_pix = retrieve_intersection_pixel(polygons, textures, intersect);
            visible[i] = true;
        }
    }

    return visible;
}

Vector3D shade_pixel(const Pixel& pix,
                     const Vector3D& norm,
                     const Vector3D& intersect_pos,
                     const std::vector<Polygon>& polygons,
                     const std::vector<Texture>& textures,
                     const std::vector<Light>& lights) {
    Vector3D colors;

    std::vector<char> visible = check_visible_lights(polygons, textures, lights, intersect_pos);
    for (int l = 0; l < visible.size(); ++l) {
        if (!visible[l]) {
            continue;
        }

        Vector3D light_dir = (lights[l].pos - intersect_pos).get_norm();
        float strength = std::max(Vector3D::dot(norm, light_dir), 0.0f);

        if (strength > 1e-5) {
            Vector3D temp;
            temp.set_x(pix.r);
            temp.set_y(pix.g);
            temp.set_z(pix.b);
            colors += strength * lights[l].color * temp;
        }
    }

    return colors;
}


Vector3D cast_ray(const std::vector<Polygon>& polygons,
                  const std::vector<Light>& lights,
                  const std::vector<Texture>& textures,
                  const Ray& ray) {
    IntersectType intersect = intersect_polygons(polygons, ray);
    const float t = intersect.first[0];
    const int polygon_id = intersect.second;
    Vector3D color;

    if (polygon_id != -1) {
        Vector3D intersect_pos = ray.pos + t * ray.dir;

        Pixel pix = retrieve_intersection_pixel(polygons, textures, intersect);
        
        color += shade_pixel(pix, polygons[polygon_id].norm(),
                             intersect_pos, polygons, textures, lights);

        color = color * (1 - pix.refract);

        // Use small value just in case for now
        if (pix.reflect > 0.05f) {
            const Vector3D& norm = polygons[polygon_id].norm();
            const Vector3D reflect_vec = Vector3D::reflect(ray.dir, norm);
            Ray reflect_ray;
            reflect_ray.pos = intersect_pos + 0.05 * norm;
            reflect_ray.dir = reflect_vec;
            color += pix.reflect * cast_ray(
                    polygons, lights, textures, reflect_ray);
        }
        if (pix.refract > 0.05f) {
            // Move forward a bit, past polygon
            Ray refract_ray;
            refract_ray.dir = ray.dir;
            refract_ray.pos = ray.pos + 0.15 * ray.dir;
            color += pix.refract * cast_ray(
                    polygons, lights, textures, refract_ray);
        }
    }

    return color;
}

void render(const std::vector<Polygon>& polygons,
            const std::vector<Light>& lights,
            const std::vector<Texture>& textures,
            const Vector3D& pc,
            const Vector3D& pv,
            int w, int h,
            float angle,
            TGAImage& image) {
    float dw = 2.0f / (w - 1);
    float dh = 2.0f / (h - 1);
    float z = 1.0f / tan(angle * M_PI / 360.0f);
    
    Matrix4x4 look_matr;
    look_matr.look_at(pv, pc, Vector3D(0.0f, 1.0f, 0.0f));

    for (int i = 0; i < w; ++i) {
        for (int j = 0; j < h; ++j) {
            Vector3D v = Vector3D(-1.0f + dw * i, (-1.0f + dh * j) * h / w, z);
            Vector3D dir = (v * look_matr).get_norm();
            Ray ray;
            ray.pos = pc;
            ray.dir = dir;

            Vector3D color_vec = cast_ray(polygons, lights, textures, ray);

            TGAColor color(
                    std::min(color_vec.x(), 255.0f),
                    std::min(color_vec.y(), 255.0f),
                    std::min(color_vec.z(), 255.0f)
                );
            image.set(i, j, color);
        }
    }
}

int main() {
    int w = 640;
    int h = 480;
    char buff[256];
    TGAImage image(w, h, TGAImage::Format::RGBA);

    std::vector<Polygon> polygons;
    std::vector<Texture> textures;
    std::vector<Light> lights;
    build_space(polygons, lights, textures);

    for (int k = 0; k < 1; ++k) {
        Vector3D pc = Vector3D(6.0f, 6.0f, 6.0f);
        Vector3D pv = Vector3D(0.0f, 0.0f, 0.0f);
        render(polygons, lights, textures, pc, pv, w, h, 120.0, image);

        sprintf(buff, "res/%d.tga", k);
        image.write_tga_file(buff, true, false);
        printf("%d: %s\n", k, buff);        
    }
}
