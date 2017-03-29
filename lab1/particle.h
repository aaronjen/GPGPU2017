#include <cmath>
#include <iostream>

class Particle{
    float x = 0;
    float y = 0;
    float vX = 0;
    float vY = 0;
    float aX = 0;
    float aY = 0;
    float maxspeed = 1;
public:
    Particle() {}
    void setPosition(unsigned x, unsigned y){
        this->x = x;
        this->y = y;
    }

    unsigned getX(){
        return round(this->x);
    }

    unsigned getY(){
        return round(this->y);
    }

    void update(){
        vX += aX;
        vY += aY;
        float speed = pow(vX, 2) + pow(vY, 2);
        speed = pow(speed, 0.5);

        if (speed > maxspeed){
            vX = vX / speed * maxspeed;
            vY = vY / speed * maxspeed;
        }
        x += vX;
        y += vY;
        aX = 0;
        aY = 0;
        checkBound();
    }
    void checkBound(){
        if (x < 0) x += 640;
        if (x >= 640) x -= 640;
        if (y < 0) y += 480;
        if (y >= 480) y -= 480;
    }

    void applyForce(float fX, float fY){
        aX += fX;
        aY += fY;
    }
};