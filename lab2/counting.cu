#include "counting.h"
#include <cstdio>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>


void printCharArray(const char* text, int size){
    char* temp = new char[size];
    cudaMemcpy(temp, text, size, cudaMemcpyDeviceToHost);
    for(int i =0; i < size; ++i){
        std::cout << temp[i];
    }
    std::cout << std::endl;
    delete[] temp;
}

void printIntArray(const int* _o, int size){
    int* temp = new int[size];
    cudaMemcpy(temp, _o, sizeof(int) * size, cudaMemcpyDeviceToHost);
    for(int i =0; i < size; ++i){
        std::cout << temp[i] << ' ';
    }
    std::cout << std::endl;
    delete[] temp;
}

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

struct isAlpha {
    __host__ __device__
    int operator()(const char& x) const { return x != '\n'; }
};

void CountPosition1(const char *text, int *pos, int text_size)
{
    isAlpha op;
    const thrust::device_ptr<const char> d_text = thrust::device_pointer_cast(text);
    thrust::device_ptr<int> d_pos = thrust::device_pointer_cast(pos);

    thrust::transform(thrust::device, d_text, d_text+text_size, d_pos, op);
    thrust::inclusive_scan_by_key(thrust::device, d_pos, d_pos+text_size, d_pos, d_pos);
}

// PART II
#define BLOCKSIZE 100

__global__ void mapping(const char* text, int* pos, int text_size){
    const int index = blockIdx.x *blockDim.x + threadIdx.x;
    if (index < text_size){
        pos[index] = text[index] != '\n';
    }
}

__global__ void upSweep(int* pos, int* key, int step, int text_size, int n_op){
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= n_op) return;
    const int indLeft = index * step * 2 + (step-1);
    const int indRight = indLeft + step;
    if(indLeft >= text_size || indRight >= text_size) return;

    const int keyLeft = key[indLeft];
    const int keyRight = key[indRight];
    const int left = pos[indLeft];
    const int right = pos[indRight];


    if(keyRight == 0){
        pos[indRight] = right;
    }
    else if(keyLeft == 0){
        pos[indRight] = right + left;
        key[indRight] = 0;
    }
    else {
        pos[indRight] = left + right;    
    }
}

__global__ void downSweep(int* pos, int* key, int step, int n_op){
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= n_op) return;
    const int indLeft = index*step*2 + (step-1);
    const int indRight = indLeft + step;

    int keyLeft = key[indLeft];
    int keyRight = key[indRight];
    int left = pos[indLeft];
    int right = pos[indRight];

    if(keyLeft == 0){
        pos[indLeft] = right;
        pos[indRight] = left;
    }
    else{
        pos[indLeft] = right;
        pos[indRight] = left + right;
    }


}

void scan(int* pos, int text_size){
    int* key;

    const size_t _s = sizeof(int)*text_size;
    cudaMalloc(&key, _s);
    cudaMemset(key, 0, _s);
    cudaMemcpy(key, pos, _s, cudaMemcpyDeviceToDevice);
    
    int _step = 1;

    while(_step*2 <= text_size){
        int n_op = CeilDiv(text_size, _step*2);
        upSweep<<<CeilDiv(n_op, BLOCKSIZE), BLOCKSIZE>>>(pos, key, _step, text_size, n_op);
        _step *= 2;
    }
    if(_step == text_size) _step /= 2;

    int last;
    int* tmp;
    const int _size = (_step == text_size? _step:_step*2);
    cudaMalloc(&tmp, sizeof(int)*_size);
    cudaMemset(tmp, 0, sizeof(int)*_size);
    cudaMemcpy(tmp, pos, sizeof(int)*text_size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(&last, tmp+_size-1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemset(tmp+_size-1, 0, sizeof(int));


    while(_step >= 1){
        int n_op = CeilDiv(_size, _step*2);
        downSweep<<<CeilDiv(n_op, BLOCKSIZE), BLOCKSIZE>>>(tmp, key,  _step, n_op);
        _step /= 2;
    }
    
    if(_size == text_size){
        cudaMemcpy(pos, tmp+1, sizeof(int)*(text_size-1), cudaMemcpyDeviceToDevice);
    }
    else{
        cudaMemcpy(pos, tmp+1, sizeof(int)*text_size, cudaMemcpyDeviceToDevice);
    }  

    cudaFree(key);
    cudaFree(tmp);
}

void CountPosition2(const char *text, int *pos, int text_size)
{
    mapping<<<text_size/BLOCKSIZE + 1, BLOCKSIZE>>>(text, pos, text_size);
    scan(pos, text_size);
}
