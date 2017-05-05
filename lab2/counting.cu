#include "counting.h"
#include <cstdio>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

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
#define BLOCKSIZE 512

__global__ void mapping(const char* text, int* pos, int text_size){
    const int index = blockIdx.x *blockDim.x + threadIdx.x;
    if (index >= text_size) return;
    pos[index] = text[index] != '\n';
    
}

__global__ void upSweep(int* pos, int* key, int step, int text_size, int n_op){
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= n_op) return;
    const int indLeft = index * step * 2 + (step-1);
    const int indRight = indLeft + step;
    if(indLeft >= text_size || indRight >= text_size) return;

    if(key[indRight] != 0){
        if(key[indLeft] == 0) key[indRight] = 0;
        pos[indRight] = pos[indLeft] + pos[indRight];
    }
}

__global__ void downSweep(int* pos, int* key, int step, int n_op){
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= n_op) return;
    const int indLeft = index*step*2 + (step-1);
    const int indRight = indLeft + step;

    const int keyLeft = key[indLeft];
    const int left = pos[indLeft];
    const int right = pos[indRight];

    if(keyLeft == 0){
        pos[indRight] = left;
    }
    else{
        pos[indRight] = left + right;
    }
    pos[indLeft] = right;
}

void scan(int* pos, int text_size){
    int full_size = 1;
    while(full_size < text_size){
        full_size *= 2;
    }
    
    int* tmp;
    const int _s = sizeof(int)*full_size;
    cudaMalloc(&tmp, _s);
    cudaMemset(tmp, 0, _s);
    cudaMemcpy(tmp, pos, sizeof(int)*text_size, cudaMemcpyDeviceToDevice);

    int reduce_step = 2;
    int n_op = 0;
    while(full_size != reduce_step){
        n_op = CeilDiv(text_size, reduce_step);
        upSweep<<<CeilDiv(n_op, BLOCKSIZE), BLOCKSIZE>>>(tmp, pos, reduce_step/2, text_size, n_op);
        reduce_step *= 2;
    }
    n_op = CeilDiv(text_size, reduce_step);
    upSweep<<<CeilDiv(n_op, BLOCKSIZE), BLOCKSIZE>>>(tmp, pos, reduce_step/2, text_size, n_op);

    int last;
    cudaMemcpy(&last, tmp+full_size-1, sizeof(int), cudaMemcpyDeviceToHost);

    while(reduce_step >= 4){
        n_op = CeilDiv(full_size, reduce_step);
        downSweep<<<CeilDiv(n_op, BLOCKSIZE), BLOCKSIZE>>>(tmp, pos, reduce_step/2, n_op);
        reduce_step /= 2;
    }
    n_op = CeilDiv(text_size, reduce_step)+1;
    downSweep<<<CeilDiv(n_op, BLOCKSIZE), BLOCKSIZE>>>(tmp, pos, reduce_step/2, n_op);

    if(full_size == text_size){
        cudaMemcpy(pos, tmp+1, sizeof(int)*(text_size-1), cudaMemcpyDeviceToDevice);
        cudaMemset(pos+text_size-1, last, sizeof(int));
    }
    else cudaMemcpy(pos, tmp+1, sizeof(int)*text_size, cudaMemcpyDeviceToDevice);

    cudaFree(tmp);
}

void CountPosition2(const char *text, int *pos, int text_size)
{
    mapping<<<text_size/BLOCKSIZE + 1, BLOCKSIZE>>>(text, pos, text_size);
    scan(pos, text_size);
}
