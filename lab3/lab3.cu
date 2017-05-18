#include "lab3.h"
#include <cstdio>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

__global__ void SimpleClone(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	if (yt < ht and xt < wt and mask[curt] > 127.0f) {
		const int yb = oy+yt, xb = ox+xt;
		const int curb = wb*yb+xb;
		if (0 <= yb and yb < hb and 0 <= xb and xb < wb) {
			output[curb*3+0] = target[curt*3+0];
			output[curb*3+1] = target[curt*3+1];
			output[curb*3+2] = target[curt*3+2];
		}
	}
}

__device__ int Clip(int v) {
	if(v < 0) return 0;
	if(v > 255) return 255;
	return v;
}

__global__ void Jacobi(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	
	const int max_t = wt*ht;
	const int max_b = wb*hb;
	if (yt < ht and xt < wt and mask[curt] > 127.0f) {
		const int yb = oy+yt, xb = ox+xt;
		const int curb = wb*yb+xb;
		if (0 <= yb and yb < hb and 0 <= xb and xb < wb) {
			// cb = 1/4(4*ct-(Nt +Wt +St +Et)+(Nb +Wb)+(Sb +Eb))
			const int nt = curt-wt;
			const int st = curt+wt;
			const int wt = curt-1;
			const int et = curt+1;

			const int nb = curb-wb;
			const int sb = curb+wb;
			const int wb = curb-1;
			const int eb = curb+1;

			const int tvs[] = {nt, st, wt, et};
			const int bvs[] = {nb, sb, wb, eb};

			int count = 0;
			int tmp[3] = {0, 0, 0};
			bool boundary = false;
			int b_diff = 0;
			for(int i = 0; i < 4; ++i){
				const int tp = tvs[i];
				const int bp = bvs[i];
				if(mask[tp] < 127 || tp < 0 || tp >= max_t){
					boundary = true;
					break;
				}

				if(bp >= 0 && bp < max_b){
					for(int j = 0; j < 3; ++j){
						const int ctv = target[curt*3+j];
						const int tpv = target[tp*3+j];
						const int bpv = output[bp*3+j];
						
						int diff = output[bp*3+j] - background[bp*3+j];
						b_diff += (diff < 0? -1 * diff: diff);
						tmp[j] += ctv - tpv + bpv;
					}
					++count;
				}
			}
			if(count && b_diff/count < 20){
				boundary = true;
			}

			__syncthreads();
			if (boundary){
				output[curb*3+0] = background[curb*3+0];
				output[curb*3+1] = background[curb*3+1];
				output[curb*3+2] = background[curb*3+2];
			}
			else if(count > 0){
				output[curb*3+0] = Clip(tmp[0] / count);
				output[curb*3+1] = Clip(tmp[1] / count);
				output[curb*3+2] = Clip(tmp[2] / count);
			}
			__syncthreads();
		}
	}
}

__global__ void CalculateFixed(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
){
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	if (yt < ht and xt < wt) {
		const int max_t = wt*ht;

		const int nt = curt-wt;
		const int st = curt+wt;
		const int wt = curt-1;
		const int et = curt+1;

		const int tvs[] = {nt, st, wt, et};
		bool boundary = false;
		for(int i=0; i< 4; ++i){
			const int p = tvs[i];
			if(mask[p] <= 127.0f || p < 0 || p >= max_t){
				boundary = true;
				break;
			}
		}

		if(mask[curt] <= 127.0f || boundary){
			const int yb = oy+yt, xb = ox+xt;
			const int curb = wb*yb+xb;
			if (0 <= yb and yb < hb and 0 <= xb and xb < wb) {
				output[curt*3+0] = background[curb*3+0];
				output[curt*3+1] = background[curb*3+1];
				output[curt*3+2] = background[curb*3+2];
			}
		}
		else {
			output[curt*3+0] = target[curt*3+0];
			output[curt*3+1] = target[curt*3+1];
			output[curt*3+2] = target[curt*3+2];
		}
	}
}

__global__ void PoissonImageCloningIteration(
	const float* fixed, 
	const float* mask, 
	const float* buf1, 
	float* output, 
	const int wt, const int ht
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	const int max_t = wt*ht;

	if (yt < ht and xt < wt and mask[curt] > 127.0f) {
		// cb = 1/4(4*ct-(Nt +Wt +St +Et)+(Nb +Wb)+(Sb +Eb))
		const int nt = curt-wt;
		const int st = curt+wt;
		const int wt = curt-1;
		const int et = curt+1;

		const int tvs[] = {nt, st, wt, et};

		int count = 0;
		int tmp[3] = {0, 0, 0};
		bool boundary = false;
		for(int i = 0; i < 4; ++i){
			const int tp = tvs[i];
			if(mask[tp] <= 127.0f){
				boundary = true;
				break;
			}

			if(tp >= 0 && tp < max_t){
				for(int j = 0; j < 3; ++j){
					const int ctv = fixed[curt*3+j];
					const int p = tp*3+j;

					const int tpv = fixed[p];
					const int bpv = buf1[p];
					
					tmp[j] += ctv - tpv + bpv;
				}
				++count;
			}	
		}
		if (boundary){
			output[curt*3+0] = fixed[curt*3+0];
			output[curt*3+1] = fixed[curt*3+1];
			output[curt*3+2] = fixed[curt*3+2];
		}
		else if(count > 0){
			output[curt*3+0] = Clip(tmp[0] / count);
			output[curt*3+1] = Clip(tmp[1] / count);
			output[curt*3+2] = Clip(tmp[2] / count);
		}
		
	}
}

void PoissonImageCloning(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
	{
	// // set up
	// float *fixed, *buf1, *buf2;
	// cudaMalloc(&fixed, 3*wt*ht*sizeof(float));
	// cudaMalloc(&buf1, 3*wt*ht*sizeof(float));
	// cudaMalloc(&buf2, 3*wt*ht*sizeof(float));
	// // initialize the iteration
	// dim3 gdim(CeilDiv(wt,32), CeilDiv(ht,16)), bdim(32,16);
	// CalculateFixed<<<gdim, bdim>>>(
	// 	background, target, mask, fixed,
	// 	wb, hb, wt, ht, oy, ox
	// );
	// cudaMemcpy(buf1, target, sizeof(float)*3*wt*ht, cudaMemcpyDeviceToDevice);

	// // iterate
	// for(int i=0;i<10000;++i){ 
	// 	PoissonImageCloningIteration<<<gdim, bdim>>>(
	// 		fixed, mask, buf1, buf2, wt, ht
	// 	);
	// 	PoissonImageCloningIteration<<<gdim, bdim>>>(
	// 		fixed, mask, buf2, buf1, wt, ht
	// 	);
	// }

	// // copy the image back
	

	// // clean up
	// cudaFree(fixed);
	// cudaFree(buf1);
	// cudaFree(buf2);


	dim3 gdim(CeilDiv(wt,32), CeilDiv(ht,16)), bdim(32,16);
	cudaMemcpy(output, background, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);
	SimpleClone<<<gdim, bdim>>>(
		background, target, mask, output,
		wb, hb, wt, ht, oy, ox
	);
	for(int i=0;i<2000;++i){
		Jacobi<<<gdim, bdim>>>(
			background, target, mask, output,
			wb, hb, wt, ht, oy, ox
		);
	}

	
}
