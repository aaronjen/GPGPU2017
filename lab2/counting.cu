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
#define BLOCKSIZE 128

__global__ void mapping(const char* text, int* pos, int text_size){
    const int index = blockIdx.x *blockDim.x + threadIdx.x;
    if (index >= text_size) return;
    pos[index] = text[index] != '\n';
    
}

__global__ void upSweep(int* pos, int* key, int step, int size){
    const int tid = threadIdx.x;
    const int sid = blockIdx.x * blockDim.x;
    if(sid+tid >= size/step) return;
    const int index = (sid+tid+1)*step-1;
    if(index >= size) return;

    __shared__ int posShared[BLOCKSIZE];
    __shared__ int keyShared[BLOCKSIZE];

    posShared[tid] = pos[index];
    keyShared[tid] = key[index];
    __syncthreads();

    for(int reduce_step = 2; reduce_step <= BLOCKSIZE; reduce_step *= 2){
        if (tid % reduce_step != (reduce_step-1)) break;
        int leftid = tid-reduce_step/2;
        if(keyShared[tid] != 0){
            if(keyShared[leftid] == 0) keyShared[tid] = 0;
            posShared[tid] = posShared[tid] + posShared[leftid];
        }
        __syncthreads();
    }

    pos[index] = posShared[tid];
    key[index] = keyShared[tid];
}

__global__ void downSweep(int* pos, int* key, int step, int size, int text_size){
    const int tid = threadIdx.x;
    const int sid = blockIdx.x * blockDim.x;
    int m_step = CeilDiv(step, BLOCKSIZE);
    if(sid+tid >= size/m_step) return;
    const int index = (sid+tid+1)*m_step-1;
    if(index >= size || index < 0) return;

    __shared__ int posShared[BLOCKSIZE];
    __shared__ int keyShared[BLOCKSIZE];

    posShared[tid] = pos[index];
    if(index >= text_size)
        keyShared[tid] = 0;
    else
        keyShared[tid] = key[index];
    __syncthreads();

    for(int s = step/m_step; s >= 2; s /= 2){
        if(tid % s == s-1){
            int leftid = tid - s/2;
            
            int right = posShared[tid];


            if(keyShared[leftid] == 0) 
                posShared[tid] = posShared[leftid];
            else
                posShared[tid] += posShared[leftid];
            posShared[leftid] = right;
        }
        __syncthreads();
    }

    pos[index] = posShared[tid];
    key[index] = keyShared[tid];
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

    int reduce_step = BLOCKSIZE;
    while(full_size > reduce_step){
        upSweep<<<CeilDiv(text_size, reduce_step), BLOCKSIZE>>>(tmp, pos, reduce_step/BLOCKSIZE, text_size);
        reduce_step *= BLOCKSIZE;
    }
    upSweep<<<CeilDiv(text_size, reduce_step), BLOCKSIZE>>>(tmp, pos, reduce_step/BLOCKSIZE, text_size);

    int last;
    cudaMemcpy(&last, tmp+full_size-1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemset(tmp+full_size-1, 0, sizeof(int));

    reduce_step = full_size;
    while(reduce_step >= 2){
        downSweep<<<CeilDiv(full_size, reduce_step), BLOCKSIZE>>>(tmp, pos, reduce_step, full_size, text_size);
        reduce_step = reduce_step/BLOCKSIZE;
    }

    if(full_size == text_size){
        cudaMemcpy(pos, tmp+1, sizeof(int)*(text_size-1), cudaMemcpyDeviceToDevice);
        cudaMemcpy(pos+text_size-1, &last, sizeof(int), cudaMemcpyHostToDevice);
    }
    else cudaMemcpy(pos, tmp+1, sizeof(int)*text_size, cudaMemcpyDeviceToDevice);
    cudaFree(tmp);
}

void CountPosition2(const char *text, int *pos, int text_size)
{
    mapping<<<CeilDiv(text_size, BLOCKSIZE), BLOCKSIZE>>>(text, pos, text_size);
    scan(pos, text_size);
}
