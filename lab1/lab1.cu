#include <cmath>
#include "lab1.h"
#include "PerlinNoise.h"
#include "particle.h"
#include <iostream>

static const unsigned W = 640;
static const unsigned H = 480;
static const unsigned NFRAME = 960;

struct Lab1VideoGenerator::Impl {
	int t = 0;
};

Lab1VideoGenerator::Lab1VideoGenerator(): impl(new Impl) {
}

Lab1VideoGenerator::~Lab1VideoGenerator() {}

void Lab1VideoGenerator::get_info(Lab1VideoInfo &info) {
	info.w = W;
	info.h = H;
	info.n_frame = NFRAME;
	// fps = 24/1 = 24
	info.fps_n = 48;
	info.fps_d = 1;
};

struct RGB {
    uint8_t r = 0;
    uint8_t g = 0;
    uint8_t b = 0;
};


/*
y = 0.299r + 0.587g + 0.114b
u = -0.169r - 0.331g + 0.5b + 128
v = 0.5r - 0.419g - 0.081b + 128
*/
uint8_t* mapRGB2YUV(RGB* arr){
    uint8_t* yuvArr = new uint8_t[W*H*3/2];

    for(int i = 0; i < W*H; ++i){
        RGB rgb = arr[i];
        uint8_t r = rgb.r;
        uint8_t g = rgb.g;
        uint8_t b = rgb.b;
        yuvArr[i] = 0.299 * r + 0.587 * g + 0.114 * b;
    }

    for(int i = W*H; i < W*H*3/2; ++i){
        yuvArr[i] = 128;
    }
    return yuvArr;
}

int particle_num = 1000;
unsigned int scl = 10;
double PI = 3.14159265359;
Particle* particles = new Particle[particle_num];
double* flowfield = new double[H/scl * W/scl];
PerlinNoise pn(222);
uint8_t* yuvArr = new uint8_t[W*H];
void Lab1VideoGenerator::Generate(uint8_t *yuv) {
    if(impl->t == 0){
        for (int i = 0; i < W*H; ++i){
            yuvArr[i] = 255;
        }
        cudaMemset(yuv+W*H, 128, W*H/2);
        for (int i = 0; i < particle_num; ++i){
            particles[i].setPosition(rand() % W, rand() % H);
        }
    }

    // flowfield
    unsigned int kk = 0;
    unsigned t = impl->t;
    for(unsigned int i = 0; i < H/scl; ++i) {
        for(unsigned int j = 0; j < W/scl; ++j) {
            double x = (double)j/W*scl;
            double y = (double)i/H*scl;

            // Typical Perlin noise
            double n = pn.noise(x, y, 0.0005 * (t+1));
            flowfield[kk] = n * 8 * PI;
            ++kk;
        }
    }

    for (int i = 0; i < particle_num; ++i){
        int x = particles[i].getX();
        int y = particles[i].getY();
        int kk = y*W + x;
        // get force
        int fx = x / scl;
        int fy = y / scl;
        int fkk = fy * (W / scl) + fx;
        double angle = flowfield[fkk];
        double mag = 5;
        double forceX = cos(angle) * mag;
        double forceY = sin(angle) * mag;

        particles[i].applyForce(forceX, forceY);
        particles[i].update();


        if(yuvArr[kk] < 5) yuvArr[kk] = 0;
        else yuvArr[kk] -= 5;
        if(yuvArr[(kk+1) % (W*H)] < 5) yuvArr[(kk+1) % (W*H)] = 0;
        else yuvArr[(kk+1) % (W*H)] -= 5;
        if(yuvArr[(kk-1) % (W*H)] < 5) yuvArr[(kk-1) % (W*H)] = 0;
        else yuvArr[(kk-1) % (W*H)] -= 5;
        if(yuvArr[(kk+W) % (W*H)] < 5) yuvArr[(kk+W) % (W*H)] = 0;
        else yuvArr[(kk+W) % (W*H)] -= 5;
        if(yuvArr[(kk-W) % (W*H)] < 5) yuvArr[(kk-W) % (W*H)] = 0;
        else yuvArr[(kk-W) % (W*H)] -= 5;
    }

    cudaMemcpy(yuv, yuvArr, W*H, cudaMemcpyHostToDevice);
	++(impl->t);
}
