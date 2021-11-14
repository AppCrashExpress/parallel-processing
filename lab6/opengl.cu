#include <iostream>
#include <math.h>
#include <random>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#define ESC 27
#define SPACEBAR 32

#define sqr3(x) ((x)*(x)*(x))
#define sqr(x)  ((x)*(x))

typedef unsigned char uchar;

struct Particle {
    float x;
    float y;
    float z;

    float dx;
    float dy;
    float dz;

    float q;
};

struct Player {
    Player() {
        x = -1.5;
        y = -1.5;
        z = 1.0;
    }

    float x;
    float y;
    float z;

    float dx;
    float dy;
    float dz;

    float yaw;
    float pitch;

    float dyaw;
    float dpitch;

    const float top_speed = 0.1;
};

namespace {
    int w = 1024;
    int h = 648;

    bool keystates[256] = {};

    const unsigned int particle_count = 4;
    const unsigned int floor_percision = 100;
    const float half_len = 15.0; // Half the length of cube edge

    cudaGraphicsResource *res;
    GLuint floor_texture;
    GLuint quad_texture;
    GLuint vbo;

    GLUquadric *quadratic;

    std::vector<Particle> particles;
    Particle *d_particles;

    Particle cam_particle;

    Player player;
}

__global__ 
void recalc_particle_velocity(Particle *particles, unsigned int count, 
                float w, float e0, float dt, float k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offsetx = blockDim.x * gridDim.x;

    for (unsigned int p = idx; p < count; p += offsetx) {
        Particle& part = particles[p];

        part.dx *= w;
        part.dy *= w;
        part.dz *= w;

        for (unsigned int p_other = 0; p_other < count + 2; ++p_other) {
            if (p_other == p) {
                continue;
            }
            Particle &other = particles[p_other];

            float l = sqrt(sqr(part.x - other.x) + sqr(part.y - other.y) + sqr(part.z - other.z));
            part.dx += other.q * part.q * k * (part.x - other.x) / (l * l * l + e0) * dt;
            part.dy += other.q * part.q * k * (part.y - other.y) / (l * l * l + e0) * dt;
            part.dz += other.q * part.q * k * (part.z - other.z) / (l * l * l + e0) * dt;
        }

        part.dx += part.q*part.q * k * (part.x - half_len) / (sqr3(fabs(part.x - half_len)) + e0) * dt;
        part.dx += part.q*part.q * k * (part.x + half_len) / (sqr3(fabs(part.x + half_len)) + e0) * dt;

        part.dy += part.q*part.q * k * (part.y - half_len) / (sqr3(fabs(part.y - half_len)) + e0) * dt;
        part.dy += part.q*part.q * k * (part.y + half_len) / (sqr3(fabs(part.y + half_len)) + e0) * dt;

        part.dz += part.q*part.q * k * (part.z - 2 * half_len) / (sqr3(fabs(part.z - 2 * half_len)) + e0) * dt;
        part.dz += part.q*part.q * k * (part.z + 0.0) / (sqr3(fabs(part.z + 0.0)) + e0) * dt;

        part.x += part.dx * dt;
        part.y += part.dy * dt;
        part.z += part.dz * dt;
    }
}

__global__
void calc_floor(uchar4 *data, Particle *particles, unsigned int count,
        float e0, float z_shift, float k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    for (int i = idx; i < floor_percision; i += offsetx) {
        for (int j = idy; j < floor_percision; j += offsety) {
            float x = (2.0 * i / (floor_percision - 1.0) - 1.0) * half_len;
            float y = (2.0 * j / (floor_percision - 1.0) - 1.0) * half_len;
            float voltage = 0;
            for (unsigned int p = 0; p < count + 1; ++p) {
                Particle &part = particles[p];
                voltage += part.q / (sqr(part.x - x) + sqr(part.y - y) + sqr(part.z - z_shift) + e0);
            }
            voltage *= k;
            data[j * floor_percision + i] = make_uchar4(min((int)voltage, 255), 0, 0, 255);
        }
    }
}

void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    gluPerspective(90.0, (GLfloat)w/(GLfloat)h, 0.1, 100.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    float cos_pitch = cos(player.pitch);
    gluLookAt(player.x, player.y, player.z,
              player.x + cos(player.yaw) * cos_pitch,
              player.y + sin(player.yaw) * cos_pitch,
              player.z + sin(player.pitch),
              0.0f, 0.0f, 1.0f);

    glBindTexture(GL_TEXTURE_2D, quad_texture);
    static float angle = 0.0;
    for (const auto& p : particles) {
        glPushMatrix();
            glTranslatef(p.x, p.y, p.z); 
            glRotatef(angle, 0.0, 0.0, 1.0);
            gluSphere(quadratic, 0.625f, 8, 8);
        glPopMatrix();
    }
    glPushMatrix();
        glTranslatef(cam_particle.x, cam_particle.y, cam_particle.z); 
        glRotatef(angle, 0.0, 0.0, 1.0);
        gluSphere(quadratic, 0.625f, 8, 8);
    glPopMatrix();
    angle += 0.15;

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, vbo);
    glBindTexture(GL_TEXTURE_2D, floor_texture);
    glTexImage2D(GL_TEXTURE_2D, 0, 3, (GLsizei)floor_percision, (GLsizei)floor_percision, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    glBegin(GL_QUADS);
        glTexCoord2f(0.0, 0.0);
        glVertex3f(-half_len, -half_len, 0.0);

        glTexCoord2f(1.0, 0.0);
        glVertex3f(half_len, -half_len, 0.0);

        glTexCoord2f(1.0, 1.0);
        glVertex3f(half_len, half_len, 0.0);

        glTexCoord2f(0.0, 1.0);
        glVertex3f(-half_len, half_len, 0.0);
    glEnd();

    glBindTexture(GL_TEXTURE_2D, 0);

    glLineWidth(2);
    glColor3f(0.5f, 0.5f, 0.5f);
    glBegin(GL_LINES);
        glVertex3f(-half_len, -half_len, 0.0);
        glVertex3f(-half_len, -half_len, 2.0 * half_len);

        glVertex3f(half_len, -half_len, 0.0);
        glVertex3f(half_len, -half_len, 2.0 * half_len);

        glVertex3f(half_len, half_len, 0.0);
        glVertex3f(half_len, half_len, 2.0 * half_len);

        glVertex3f(-half_len, half_len, 0.0);
        glVertex3f(-half_len, half_len, 2.0 * half_len);
    glEnd();

    glBegin(GL_LINE_LOOP);
        glVertex3f(-half_len, -half_len, 0.0);
        glVertex3f( half_len, -half_len, 0.0);
        glVertex3f( half_len,  half_len, 0.0);
        glVertex3f(-half_len,  half_len, 0.0);
    glEnd();

    glBegin(GL_LINE_LOOP);
        glVertex3f(-half_len, -half_len, 2.0 * half_len);
        glVertex3f( half_len, -half_len, 2.0 * half_len);
        glVertex3f( half_len,  half_len, 2.0 * half_len);
        glVertex3f(-half_len,  half_len, 2.0 * half_len);
    glEnd();

    glColor3f(1.0f, 1.0f, 1.0f);

    glutSwapBuffers();
}

void key_down(unsigned char key, int x, int y) {
    keystates[key] = true;
}

void key_up(unsigned char key, int x, int y) {
    keystates[key] = false;
}

void process_keys() {
    float top_speed = player.top_speed;

    if (keystates['w']) {
        float cos_pitch = cos(player.pitch);
        player.dx += cos(player.yaw) * cos_pitch * top_speed;
        player.dy += sin(player.yaw) * cos_pitch * top_speed;
        player.dz += sin(player.pitch) * top_speed;
    }
    if (keystates['s']) {
        float cos_pitch = cos(player.pitch);
        player.dx -= cos(player.yaw) * cos_pitch * top_speed;
        player.dy -= sin(player.yaw) * cos_pitch * top_speed;
        player.dz -= sin(player.pitch) * top_speed;
    }
    if (keystates['a']) {
        player.dx += -sin(player.yaw) * top_speed;
        player.dy += cos(player.yaw) * top_speed;
    }
    if (keystates['d']) {
        player.dx += sin(player.yaw) * top_speed;
        player.dy += -cos(player.yaw) * top_speed;
    }
    if (keystates[SPACEBAR]) {
        player.dx = 0;
        player.dy = 0;
        player.dz = 0;
    }
    if (keystates[ESC]) {
        cudaGraphicsUnregisterResource(res);
        glDeleteTextures(1, &floor_texture);
        glDeleteTextures(1, &quad_texture);
        glDeleteBuffers(1, &vbo);
        gluDeleteQuadric(quadratic);
        cudaFree(d_particles);
        exit(0);
    }
}

void init_cam_particle() {
    cam_particle.x = 0.0f;
    cam_particle.y = 0.0f;
    cam_particle.z = -200.0f;

    cam_particle.dx = 0;
    cam_particle.dy = 0;
    cam_particle.dz = 0;
}

void process_cam_particle(float dt) {
    cam_particle.x += cam_particle.dx * dt;
    cam_particle.y += cam_particle.dy * dt;
    cam_particle.z += cam_particle.dz * dt;

    float box_limit = half_len + 20.0;
    if (abs(cam_particle.x) >= box_limit || 
            abs(cam_particle.y) >= box_limit ||
            abs(cam_particle.z) >= half_len + box_limit) {
        init_cam_particle();
    }
}

void update() {
    process_keys();

    const float speed = player.top_speed;
    float v = sqrt(player.dx * player.dx + player.dy * player.dy + player.dz * player.dz);
    if (v > speed) {
        float norm = speed / v;
        player.dx *= norm;
        player.dy *= norm;
        player.dz *= norm;
    }

    float slow_down = 0.99;
    player.x += player.dx;
    player.y += player.dy;
    player.z += player.dz;
    player.dx *= slow_down;
    player.dy *= slow_down;
    player.dz *= slow_down;

    if (player.z < 1.0) {
        player.z = 1.0;
        player.dz = 0.0;
    }

    if (fabs(player.dpitch) + fabs(player.dyaw) > 0.00001) {
        player.yaw += player.dyaw;
        player.pitch += player.dpitch;
        player.pitch = min(M_PI / 2.0 - 0.0001, max(-M_PI / 2.0 + 0.0001, player.pitch));
        player.dyaw = player.dpitch = 0.0;
    }

    float w = 0.9999, e0 = 1e-3, dt = 0.01, k = 50.0;

    process_cam_particle(dt);

    cudaMemcpy(d_particles, particles.data(), sizeof(Particle) * particles.size(), cudaMemcpyHostToDevice);
    Particle player_particle;
    player_particle.x = player.x;
    player_particle.y = player.y;
    player_particle.z = player.z;
    player_particle.q = 10;
    cudaMemcpy(d_particles + particles.size(), &cam_particle, sizeof(Particle), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles + particles.size() + 1, &player_particle, sizeof(Particle), cudaMemcpyHostToDevice);

    recalc_particle_velocity<<<256, 256>>>(d_particles, particles.size(), w, e0, dt, k);

    cudaMemcpy(particles.data(), d_particles, sizeof(Particle) * particles.size(), cudaMemcpyDeviceToHost);

    static float t = 0.0;
    uchar4 *dev_data;
    size_t size;
    cudaGraphicsMapResources(1, &res, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&dev_data, &size, res);

    calc_floor<<<dim3(32, 32), dim3(32, 8)>>>(dev_data, d_particles, particles.size(), e0, 0.75, k);

    cudaGraphicsUnmapResources(1, &res, 0);
    t += 0.01;

    glutPostRedisplay();
}

void mouse(int x, int y) {
    if (x != w/2 || y != h/2) {
        glutWarpPointer(w / 2, h / 2);
        float norm_coef = 0.001;
        float dx = norm_coef * (x - w/2);
        float dy = norm_coef * (y - h/2);
        player.dyaw -= dx;
        player.dpitch -= dy;
    }
}

void mouse_press(int button, int state, int x, int y) {
    cam_particle.x = player.x;
    cam_particle.y = player.y;
    cam_particle.z = player.z;

    float speed = 30.0;
    float cos_pitch = cos(player.pitch);
    cam_particle.dx = speed * cos(player.yaw) * cos_pitch;
    cam_particle.dy = speed * sin(player.yaw) * cos_pitch;
    cam_particle.dz = speed * sin(player.pitch);

    cam_particle.q = 50;
}

void reshape(int w_new, int h_new) {
    w = w_new;
    h = h_new;
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
}

std::vector<Particle> 
fill_with_random_particles(unsigned int particle_count) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-half_len, half_len);

    std::vector<Particle> particles(particle_count);

    for (auto& p : particles) {
        p.x = dist(gen);
        p.y = dist(gen);
        p.z = half_len + dist(gen);
        p.q = 1;
    }

    return particles;
}

int main(int argc, char *argv[]) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(w, h);
    glutCreateWindow("Particle simulator 2021");

    glutIdleFunc(update);
    glutDisplayFunc(display);
    glutKeyboardFunc(key_down);
    glutKeyboardUpFunc(key_up);
    glutPassiveMotionFunc(mouse);
    glutMouseFunc(mouse_press);
    glutReshapeFunc(reshape);

    // glutSetCursor(GLUT_CURSOR_NONE);

    int wt, ht;
    FILE *in = fopen("in.data", "rb");
    fread(&wt, sizeof(int), 1, in);
    fread(&ht, sizeof(int), 1, in);
    uchar *data = (uchar*) malloc(sizeof(uchar) * wt * ht * 4);
    fread(data, sizeof(uchar), wt * ht * 4, in);
    fclose(in);

    glGenTextures(1, &quad_texture);
    glBindTexture(GL_TEXTURE_2D, quad_texture);
    glTexImage2D(GL_TEXTURE_2D, 0, 3, (GLsizei)wt, (GLsizei)ht, 0, GL_RGBA, GL_UNSIGNED_BYTE, (void*)data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    free(data);

    quadratic = gluNewQuadric();
    gluQuadricTexture(quadratic, GL_TRUE);

    glGenTextures(1, &floor_texture);
    glBindTexture(GL_TEXTURE_2D, floor_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glEnable(GL_TEXTURE_2D);
    glShadeModel(GL_SMOOTH);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClearDepth(1.0f);
    glDepthFunc(GL_LEQUAL);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);

    glewInit();
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, vbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, 
                 floor_percision * floor_percision * sizeof(uchar4), 
                 NULL,
                 GL_DYNAMIC_DRAW);
    cudaGraphicsGLRegisterBuffer(&res, vbo, cudaGraphicsMapFlagsWriteDiscard);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    particles = fill_with_random_particles(particle_count);
    init_cam_particle();
    // Two more for camera shot particle and for player particle
    cudaMalloc(&d_particles, sizeof(Particle) * (particle_count + 2));

    glutMainLoop();
}
