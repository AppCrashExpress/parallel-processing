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

int MAX_DEEP;

void build_space(std::vector<Polygon>& polys,
        std::vector<Light>& lights,
        std::vector<Texture>& textures) {
    Pixel pixels[4] = {
        {255,   0,   0, 0.0f, 0.0f },
        {  0, 255,   0, 0.0f, 0.0f },
        {  0,   0, 255, 0.0f, 0.0f },
        {255, 255, 255, 0.0f, 0.0f }
    };

    for( int i = 0 ; i < 4 ; ++i )
        pixels[i].reflect = 1.0f;

    textures.push_back(Texture(2, 2, pixels));

    Pixel pixel = {255, 0,0, 0.0f, 0.0f};

    textures.push_back(Texture(1, 1, &pixel));

    pixel = {0, 255, 0, 0.0f, 0.0f};
    textures.push_back(Texture(1, 1, &pixel));

    pixel = {0, 0,  255, 0.0f, 0.4f};

    textures.push_back(Texture(1, 1, &pixel));

    pixel.r = 200;
    pixel.b = 200;
    pixel.g = 20;
    pixel.reflect = 0.1;
    pixel.refract = 0;
    textures.push_back(Texture(1, 1, &pixel));

    std::vector<Polygon> cube;

    cube = construct_cube(Vector3D(0, 0, 0), 1.0f, 3);
    polys.insert(polys.end(), cube.begin(), cube.end());

    std::vector<Polygon> floor = construct_floor(Vector3D(0.0f, -2.0f, 0.0f), 10, 4);
    polys.insert(polys.end(), floor.begin(), floor.end());

    const Vector3D light_color = Vector3D(1.0f, 1.0f, 1.0f);

    lights = std::vector<Light>{
        {Vector3D(-2.0f, 6.0f, -2.0f), light_color}};
}

std::pair<Vector3D, int> 
intersect_polygons(const std::vector<Polygon>& polygons,
        const Ray& ray) {
    Vector3D min_res;
    int max_poly = -1;


    min_res[0] = 1e10;
    for (int i = 0; i < polygons.size(); ++i) {
        std::pair<Vector3D, bool> res = polygons[i].find_intersection(ray, 1e-7);
        if (res.second && min_res[0] > res.first[0]) {
            min_res = res.first;
            max_poly = i;
        }
    }

    return {min_res, max_poly};
}

std::tuple<Vector3D, int, float> 
intersect_polygons_refracted(const std::vector<Polygon>& polygons,
        const std::vector<Texture>& textures,
        const Ray& ray) {
    Vector3D min_res;
    int max_poly = -1;

    float min_refracted = 1.0f, cur_refracted;

    min_res[0] = 1e10;
    for (int i = 0; i < polygons.size(); ++i) {
        std::pair<Vector3D, bool> res = polygons[i].find_intersection(ray, 1e-7);
        if (res.second && min_res[0] > res.first[0]) {
            cur_refracted = textures[polygons[i].get_texture_id()].at({0,0}).refract;
            min_refracted = std::min(min_refracted, cur_refracted);
            min_res = res.first;
            max_poly = i;
        }
    }

    return {min_res, max_poly, min_refracted};
}

Pixel 
retrieve_intersection_pixel(const std::vector<Polygon>& polygons,
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

std::vector<float> 
check_visible_lights(const std::vector<Polygon>& polygons,
        const std::vector<Texture>& textures,
        const std::vector<Light>& lights,
        const Vector3D& pos) {

    std::vector<float> visible(lights.size(), false);

    for (size_t i = 0; i < lights.size(); ++i) {
        Vector3D dir = pos - lights[i].pos;
        float t_max = dir.get_length() - 1e-4f;
        dir = dir.get_norm();

        Ray ray;
        ray.pos = lights[i].pos;
        ray.dir = dir;
        std::tuple<Vector3D, int, float>  intersect = intersect_polygons_refracted(polygons, textures, ray);

        float t = std::get<0>(intersect)[0];
        float min_refraction = std::get<2>(intersect);
        if (t < 0 || t >= t_max) {
            //Pixel hit_pix = retrieve_intersection_pixel(polygons, textures, intersect);
            visible[i] = true;
        }
        if(min_refraction > 0 && (t>1.0f || t<-1.0f))
            visible[i] = min_refraction;

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

    std::vector<float> visible = check_visible_lights(polygons, textures, lights, intersect_pos);
    for (int l = 0; l < visible.size(); ++l) {
        if (visible[l]<0.001f) {
            continue;
        }

        Vector3D light_dir = (lights[l].pos - intersect_pos).get_norm();
        float strength = std::max(Vector3D::dot(norm, light_dir), 0.0f)*visible[l];

        if (strength > 1e-5) {
            Vector3D temp;
            temp.set_x(pix.r / 255.0f);
            temp.set_y(pix.g / 255.0f);
            temp.set_z(pix.b / 255.0f);
            colors += strength * lights[l].color * temp;
        }
    }

    return colors;
}


Vector3D cast_ray(
        const std::vector<Polygon>& polygons,
        const std::vector<Light>& lights,
        const std::vector<Texture>& textures,
        const Ray& ray,
        const int deep,
        const int max_deep);

Vector3D cast_inner_ray(
        const std::vector<Polygon>& polygons,
        const std::vector<Light>& lights,
        const std::vector<Texture>& textures,
        const Ray& ray,
        const int deep,
        const int max_deep) {
    if(deep>=max_deep)
        return {0.0f, 0.0f, 0.0f};
    IntersectType intersect = intersect_polygons(polygons, ray);
    const float t = intersect.first[0];
    const int polygon_id = intersect.second;

    Vector3D color = {0.0f, 0.0f, 0.0f};
    Vector3D intersect_pos = ray.pos + t * ray.dir ;

    Pixel pix = retrieve_intersection_pixel(polygons, textures, intersect);

    const Vector3D& norm = polygons[polygon_id].norm();
    Ray reflect_ray;
    reflect_ray.pos = intersect_pos + 0.005 * norm;

    reflect_ray.dir = ray.dir;
    color += pix.refract */*(t<1.0f ? t : 1.0f/t)**/cast_ray(
            polygons, lights, textures, reflect_ray, deep+1, max_deep);



    return color;

}


Vector3D cast_ray(
        const std::vector<Polygon>& polygons,
        const std::vector<Light>& lights,
        const std::vector<Texture>& textures,
        const Ray& ray,
        const int deep,
        const int max_deep) {
    if(deep>=max_deep)
        return {0.0f, 0.0f, 0.0f};
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
            color += pix.reflect * (1.0f - pix.refract)* cast_ray(
                    polygons, lights, textures, reflect_ray, deep+1, max_deep);
        }
        if (pix.refract > 0.05f) {
            // Move forward a bit, past polygon
            Ray refract_ray;
            refract_ray.dir = ray.dir;
            refract_ray.pos = ray.pos + 0.05 * ray.dir;
            color += pix.refract * cast_inner_ray(
                    polygons, lights, textures, refract_ray, deep+1, max_deep);
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

            Vector3D color_vec = cast_ray(polygons, lights, textures, ray, 0, MAX_DEEP);

            TGAColor color(
                    std::min(color_vec.x() * 255.0f, 255.0f),
                    std::min(color_vec.y() * 255.0f, 255.0f),
                    std::min(color_vec.z() * 255.0f, 255.0f)
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
    int k;
    for ( k = 0; k < 40; k+=1) {
        MAX_DEEP = 5+k/2;
        Vector3D pc = Vector3D(5.0f + 1.0f * cos(0.3f * float(k)), 5.0f + 2.0f * sin(0.1f * float(k)), 5.0f + 2.0f * cos(0.1f * float(k)));
        Vector3D pv = Vector3D(0.0f, 0.0f, 0.0f);
        render(polygons, lights, textures, pc, pv, w, h, 90.0, image);

        sprintf(buff, "res/%03d_deep_%03d.tga", k, MAX_DEEP);
        image.write_tga_file(buff, true, false);
        printf("%d: %s\n", k, buff);    
    }
}
