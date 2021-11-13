#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

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
} item;

int w = 1024, h = 648;

float x = -1.5, y = -1.5, z = 1.0;
float dx = 0.0, dy = 0.0, dz = 0.0;
float yaw = 0.0, pitch = 0.0;
float dyaw = 0.0, dpitch = 0.0;

float speed = 0.05;

const float a2 = 15.0;
const int np = 100;

cudaGraphicsResource *res;
GLuint textures[2];
GLuint vbo;

GLUquadric* quadratic;


__global__ void kernel(uchar4 *data, Particle item, float t) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;
    int i, j;
    float x, y, fg, fb;

    for (i = idx; i < np; i += offsetx) {
        for (j = idy; j < np; j += offsety) {
            x = (2.0 * i / (np - 1.0) - 1.0) * a2;
            y = (2.0 * j / (np - 1.0) - 1.0) * a2;
            fb = 100.0 * ( sin(0.1 * x*x + t) + cos(0.1 * y*y + t * 0.6) + sin(0.1 * x*x + 0.1 * y*y + t * 0.3) );
            fg = 10000.0 * item.q / ( sqr(x - item.x) + sqr(y - item.y) + sqr(item.z) + 0.001 );
            fg = min(max(0.0f, fg), 255.0f);
            fb = min(max(0.0f, fb), 255.0f);
            data[j * np + i] = make_uchar4(0, (int)fg, (int)fb, 255);
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
    float cos_pitch = cos(pitch);
    gluLookAt(x, y, z,
              x + cos(yaw) * cos_pitch,
              y + sin(yaw) * cos_pitch,
              z + sin(pitch),
              0.0f, 0.0f, 1.0f);
    glBindTexture(GL_TEXTURE_2D, textures[0]);

    static float angle = 0.0;
    glPushMatrix();
        glTranslatef(item.x, item.y, item.z); 
        glRotatef(angle, 0.0, 0.0, 1.0);
        gluSphere(quadratic, 2.5f, 32, 32);
    glPopMatrix();
    angle += 0.15;

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, vbo);
    glBindTexture(GL_TEXTURE_2D, textures[1]);
    glTexImage2D(GL_TEXTURE_2D, 0, 3, (GLsizei)np, (GLsizei)np, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    glBegin(GL_QUADS);
        glTexCoord2f(0.0, 0.0);
        glVertex3f(-a2, -a2, 0.0);

        glTexCoord2f(1.0, 0.0);
        glVertex3f(a2, -a2, 0.0);

        glTexCoord2f(1.0, 1.0);
        glVertex3f(a2, a2, 0.0);

        glTexCoord2f(0.0, 1.0);
        glVertex3f(-a2, a2, 0.0);
    glEnd();

    glBindTexture(GL_TEXTURE_2D, 0);

    glLineWidth(2);
    glColor3f(0.5f, 0.5f, 0.5f);
    glBegin(GL_LINES);
        glVertex3f(-a2, -a2, 0.0);
        glVertex3f(-a2, -a2, 2.0 * a2);

        glVertex3f(a2, -a2, 0.0);
        glVertex3f(a2, -a2, 2.0 * a2);

        glVertex3f(a2, a2, 0.0);
        glVertex3f(a2, a2, 2.0 * a2);

        glVertex3f(-a2, a2, 0.0);
        glVertex3f(-a2, a2, 2.0 * a2);
    glEnd();

    glBegin(GL_LINE_LOOP);
        glVertex3f(-a2, -a2, 0.0);
        glVertex3f( a2, -a2, 0.0);
        glVertex3f( a2,  a2, 0.0);
        glVertex3f(-a2,  a2, 0.0);
    glEnd();

    glBegin(GL_LINE_LOOP);
        glVertex3f(-a2, -a2, 2.0 * a2);
        glVertex3f( a2, -a2, 2.0 * a2);
        glVertex3f( a2,  a2, 2.0 * a2);
        glVertex3f(-a2,  a2, 2.0 * a2);
    glEnd();

    glColor3f(1.0f, 1.0f, 1.0f);

    glutSwapBuffers();
}

void update() {
    dz -= 0.0001;
    float v = sqrt(dx * dx + dy * dy + dz * dz);
    if (v > speed) {
        float norm = speed / v;
        dx *= norm;
        dy *= norm;
        dz *= norm;
    }

    float slow_down = 0.99;
    x += dx;
    y += dy;
    z += dz;
    dx *= slow_down;
    dy *= slow_down;
    dz *= slow_down;

    if (z < 1.0) {
        z = 1.0;
        dz = 0.0;
    }

    if (fabs(dpitch) + fabs(dyaw) > 0.00001) {
        yaw += dyaw;
        pitch += dpitch;
        pitch = min(M_PI / 2.0 - 0.0001, max(-M_PI / 2.0 + 0.0001, pitch));
        dyaw = dpitch = 0.0;
    }

    float w = 0.9999, e0 = 1e-3, dt = 0.01, k = 50.0;

    item.dx *= w;
    item.dy *= w;
    item.dz *= w;

    item.dx += item.q*item.q * k * (item.x - a2) / (sqr3(fabs(item.x - a2)) + e0) * dt;
    item.dx += item.q*item.q * k * (item.x + a2) / (sqr3(fabs(item.x + a2)) + e0) * dt;

    item.dy += item.q*item.q * k * (item.y - a2) / (sqr3(fabs(item.y - a2)) + e0) * dt;
    item.dy += item.q*item.q * k * (item.y + a2) / (sqr3(fabs(item.y + a2)) + e0) * dt;

    item.dz += item.q*item.q * k * (item.z - 2 * a2) / (sqr3(fabs(item.z - 2 * a2)) + e0) * dt;
    item.dz += item.q*item.q * k * (item.z + 0.0) / (sqr3(fabs(item.z + 0.0)) + e0) * dt;

    float l = sqrt(sqr(item.x - x) + sqr(item.y - y) + sqr(item.z - z));
    item.dx += 3.0 * item.q * k * (item.x - x) / (l * l * l + e0) * dt;
    item.dy += 3.0 * item.q * k * (item.y - y) / (l * l * l + e0) * dt;
    item.dz += 3.0 * item.q * k * (item.z - z) / (l * l * l + e0) * dt;

    item.x += item.dx * dt;
    item.y += item.dy * dt;
    item.z += item.dz * dt;

    static float t = 0.0;
    uchar4 *dev_data;
    size_t size;
    cudaGraphicsMapResources(1, &res, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&dev_data, &size, res);
    kernel<<<dim3(32, 32), dim3(32, 8)>>>(dev_data, item, t);
    cudaGraphicsUnmapResources(1, &res, 0);
    t += 0.01;

    glutPostRedisplay();
}

void keys(unsigned char key, int x, int y) {
    switch(key) {
      case 'w': {
        float cos_pitch = cos(pitch);
        dx += cos(yaw) * cos_pitch * speed;
        dy += sin(yaw) * cos_pitch * speed;
        dz += sin(pitch) * speed;
        break;
      }
      case 's': {
        float cos_pitch = cos(pitch);
        dx -= cos(yaw) * cos_pitch * speed;
        dy -= sin(yaw) * cos_pitch * speed;
        dz -= sin(pitch) * speed;
        break;
      }
      case 'a': {
        dx += -sin(yaw) * speed;
        dy += cos(yaw) * speed;
        break;
      }
      case 'd': {
        dx += sin(yaw) * speed;
        dy += -cos(yaw) * speed;
        break;
      }
      case 27: {
        cudaGraphicsUnregisterResource(res);
        glDeleteTextures(2, textures);
        glDeleteBuffers(1, &vbo);
        gluDeleteQuadric(quadratic);
        exit(0);
        break;
      }
    }
}

void mouse(int x, int y) {
    static int x_prev = w / 2, y_prev = h / 2;
    float norm_coef = 0.005;
    float dx = norm_coef * (x - x_prev);
    float dy = norm_coef * (y - y_prev);

    dyaw -= dx;
    dpitch -= dy;
    x_prev = x;
    y_prev = y;

    if ((x < 20) || (y < 20) || (x > w - 20) || (y > h - 20)) {
        glutWarpPointer(w / 2, h / 2);
        x_prev = w / 2;
        y_prev = h / 2;
    }
}

void reshape(int w_new, int h_new) {
    w = w_new;
    h = h_new;
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
}

int main(int argc, char *argv[]) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(w, h);
    glutCreateWindow("Particle simulator 2021");

    glutIdleFunc(update);
    glutDisplayFunc(display);
    glutKeyboardFunc(keys);
    glutPassiveMotionFunc(mouse);
    glutReshapeFunc(reshape);

    glutSetCursor(GLUT_CURSOR_NONE);

    int wt, ht;
    FILE *in = fopen("in.data", "rb");
    fread(&wt, sizeof(int), 1, in);
    fread(&ht, sizeof(int), 1, in);
    uchar *data = (uchar*) malloc(sizeof(uchar) * wt * ht * 4);
    fread(data, sizeof(uchar), wt * ht * 4, in);
    fclose(in);

    glGenTextures(2, textures);
    glBindTexture(GL_TEXTURE_2D, textures[0]);
    glTexImage2D(GL_TEXTURE_2D, 0, 3, (GLsizei)wt, (GLsizei)ht, 0, GL_RGBA, GL_UNSIGNED_BYTE, (void*)data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    quadratic = gluNewQuadric();
    gluQuadricTexture(quadratic, GL_TRUE);

    glBindTexture(GL_TEXTURE_2D, textures[1]);
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
    glBufferData(GL_PIXEL_UNPACK_BUFFER, np * np * sizeof(uchar4), NULL, GL_DYNAMIC_DRAW);
    cudaGraphicsGLRegisterBuffer(&res, vbo, cudaGraphicsMapFlagsWriteDiscard);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    item.x = item.y = item.z = 5.0;
    item.dx = item.dy = item.dz = 0.1;
    item.q = 1.0;

    glutMainLoop();

    free(data);
}
