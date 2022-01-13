#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <omp.h>

#include "tgaimage.h"
#include "linear.h"
#include "polygon.h"
#include "ray.h"
#include "model.h"
#include "light.h"
#include "texture.h"

using IntersectType = std::pair<Vector3D, int>;


std::vector<Polygon> polygons;
std::vector<Light> lights;
std::vector<Texture> textures;

size_t polygons_size,  lights_size, textures_size;



void build_space() {
    Pixel pixels[4] = {
        {255,   0,   0, 0.7f, 0.7f,},
        {  0, 255,   0, 0.7f, 0.7f,},
        {  0,   0, 255, 0.7f, 0.7f,},
        {255, 255, 255, 0.7f, 0.7f,},
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
    pixel.reflect = 0.25;
    textures.push_back(Texture(1, 1, &pixel));

    pixel.r = 100;
    pixel.b = 100;
    pixel.g = 100;
    pixel.reflect = 1.0;
    pixel.refract = 0;
    textures.push_back(Texture(1, 1, &pixel));

    std::vector<Polygon> cube;

    cube = construct_cube(Vector3D(0, 0, 0), 1.0f, 0);
    polygons.insert(polygons.end(), cube.begin(), cube.end());

    cube = construct_cube(Vector3D(5.0f, 0, 0), 1.0f, 1);
    polygons.insert(polygons.end(), cube.begin(), cube.end());

    cube = construct_cube(Vector3D(0, 5.0f, 0), 1.0f, 2);
    polygons.insert(polygons.end(), cube.begin(), cube.end());

    cube = construct_cube(Vector3D(0, 0, 5.0f), 1.0f, 3);
    polygons.insert(polygons.end(), cube.begin(), cube.end());

    std::vector<Polygon> floor = construct_floor(Vector3D(0.0f, -2.0f, 0.0f), 10, 4);
    polygons.insert(polygons.end(), floor.begin(), floor.end());

    const Vector3D light_color = Vector3D(1.0f, 1.0f, 1.0f);

    lights = std::vector<Light>{{Vector3D(2.0f, 0.0f, 0.0f), light_color},
                                {Vector3D(0.0f, 2.0f, 0.0f), light_color}};
    
    polygons_size = polygons.size();  
    lights_size   = lights.size(); 
    textures_size = textures.size();                 
}

std::pair<Vector3D, int> intersect_polygons(const Ray& ray) {
    Vector3D max_res;
    int max_poly = -1;

    max_res[0] = 1e10;
    for (int i = 0; i < polygons_size; ++i) {
        std::pair<Vector3D, bool> res = polygons[i].find_intersection(ray, 1e-7);
        if (res.second && max_res[0] > res.first[0]) {
            max_res = res.first;
            max_poly = i;
        }
    }

    return {max_res, max_poly};
}

Pixel retrieve_intersection_pixel(const IntersectType& intersection) {

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
check_visible_lights(const Vector3D& pos) {
    std::vector<char> visible(lights_size, false);

    for (size_t i = 0; i < lights_size; ++i) {
        Vector3D dir = pos - lights[i].pos;
        float t_max = dir.get_length() - 1e-4f;
        dir = dir.get_norm();

        Ray ray;
        ray.pos = lights[i].pos;
        ray.dir = dir;
        IntersectType intersect = intersect_polygons(ray);

        float t = intersect.first[0];
        if (t < 0 || t >= t_max) {
        //    Pixel hit_pix = retrieve_intersection_pixel(intersect);
            visible[i] = true;
        }
    }

    return visible;
}

Vector3D shade_pixel(const Pixel& pix,
                     const Vector3D& norm,
                     const Vector3D& intersect_pos) {
    Vector3D colors;

    std::vector<char> visible = check_visible_lights(intersect_pos);
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


Vector3D cast_ray(Ray& ray, Ray& reflection_ray) {

    IntersectType intersect = intersect_polygons(ray);
    const float t = intersect.first[0];
    const int polygon_id = intersect.second;
    Vector3D color;

    if (polygon_id != -1) {
        Vector3D intersect_pos = ray.pos + t * ray.dir;

        Pixel pix = retrieve_intersection_pixel(intersect);
        color += shade_pixel(pix, polygons[polygon_id].norm(), intersect_pos);

        ray.color = color;

        // Use small value just in case for now
        if (pix.reflect > 0.03f) {
            const Vector3D& norm = polygons[polygon_id].norm();
            const Vector3D reflect_vec = Vector3D::reflect(ray.dir, norm);
            
            reflection_ray.pos = intersect_pos + 0.03 * norm;
            reflection_ray.dir = reflect_vec;

            reflection_ray.consid_coef =  pix.reflect * ray.consid_coef;
            reflection_ray.x = ray.x;
            reflection_ray.y = ray.y;

        }

    }else{
        //reflection_ray.use = false;
        ray.color = {0.0f, 0.0f, 0.0f};
        
    }

    return color;
}


void
do_rays(Ray* cur_rays, Ray* next_rays, size_t cur_size){
    size_t i;
    #pragma omp parallel for private(i) shared(cur_rays, next_rays)
    for(i = 0 ; i < cur_size; ++i) {
        cast_ray(cur_rays[i], next_rays[i]);
    }
}


void
do_colors(Vector3D* image_buff, Ray* rays,const int rays_size, const int h){
    int i;
    #pragma omp parallel for private(i) shared(image_buff, rays)
    for (i = 0; i < rays_size; ++i){
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

void render(const Vector3D& pc,
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


    int i, j;
    #pragma omp parallel for private(i, j) shared(cur_rays)
    for (i = 0; i < w; ++i) {
        for (j = 0; j < h; ++j) {
            Vector3D v = Vector3D(-1.0f + dw * i, (-1.0f + dh * j) * h / w, z);
            Vector3D dir = (v * look_matr).get_norm();
            cur_rays[i*h+j] = Ray(pc, dir, j, i);

        }
    }

    int cur_size = pixel_cnt;

    for (int depth = 0; depth < max_depth; ++depth) {
    
        do_rays (cur_rays.get(), next_rays.get(), cur_size);
        do_colors(image_buff.get(),cur_rays.get(), cur_size, h);
        
        cur_size = compact(next_rays.get(), cur_size);
        
        std::swap(cur_rays, next_rays);
    }

    set_image(image_buff.get(), image, w,h);
}

int main() {
    int w = 640;
    int h = 480;
    char buff[256];
    TGAImage image(w, h, TGAImage::Format::RGBA);

    build_space();

    int max_depth = 3;
    for (int k = 0; k < 10; ++k) {
        Vector3D pc = Vector3D( 2.0f +3.0f*sin(k*0.2f),  3.0f + 3.0f*cos(k*0.1f), 3.0f + 3.0f*sin(k*0.1f));
        Vector3D pv = Vector3D(0.0f, 0.0f, 0.0f);
        render(pc, pv, w, h, 120.0, image, max_depth);

        sprintf(buff, "res/%d.tga", k);
        image.write_tga_file(buff, true, false);
        printf("%d: %s\n", k, buff);        
    }
}
