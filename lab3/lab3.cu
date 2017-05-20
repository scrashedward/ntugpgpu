#include "lab3.h"
#include <cstdio>
#include <iostream>

using namespace std;

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

__global__ void CalculateFixed(
	const float *background,
	const float *target,
	const float *mask,
	float *fixed,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int x = bx*32 + tx;
	int y = by*16 + ty;
	int bgx = ox + x;
	int bgy = oy + y;
	int idx = y * wt + x;
	int bidx = bgy * wb + bgx;

	for(int i=0;i<3;i++){
		if(mask[idx] == 0){
			fixed[idx*3 + i] = background[bidx*3 + i];
			continue;
		}
		float tc = target[idx*3+i];
		float tn(tc), tw(tc), ts(tc), te(tc);
		if( x != 0){
			tw = target[(idx - 1) * 3 + i];
		}
		if( x != wt-1 ){
			te = target[(idx + 1) * 3 + i];
		}
		if( y != 0 ){
			tn = target[(idx - wt) * 3 + i];
		}
		if( y != ht-1){
			ts = target[(idx + wt) * 3 + i];
		}

		float r = 4 * tc - (tn + tw + ts + te);

		if( x == 0 || mask[idx-1] == 0){
			if(bgx == 0) r += background[bidx * 3 + i];
			else r += background[(bidx - 1) * 3 + i];
		}
		if( x == wt-1 || mask[idx+1] == 0){
			if(bgx == wb-1) r += background[bidx * 3 + i];
			else r += background[(bidx + 1) * 3 + i];
		}
		if( y == 0 || mask[idx-wt] == 0){
			if(bgy == 0 ) r += background[bidx * 3 + i];
			else r += background[(bidx - wb) * 3 + i];
		}
		if( y == ht -1 || mask[idx+wt] == 0){
			if(bgy == hb-1) r += background[bidx * 3 + i];
			else r += background[(bidx + wb) * 3 + i];
		}

		fixed[idx * 3 + i] = r;
	}
}

__global__ void PoissonImageCloningIteration(
	const float *fixed,
	const float *mask,
	float *source, float *dest, int wt, int ht
)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int x = bx * 32 + tx;
	int y = by * 16 + ty;
	int idx = y * wt + x;

	for(int i = 0; i < 3; i++){
		if (mask[idx] == 0 ){
			dest[idx * 3 + i] = fixed[idx * 3 + i];
			continue;
		}
		float r = fixed[idx * 3 + i];
		if ( x != 0 && mask[idx-1] != 0 ){
			r += source[(idx - 1)* 3+ i];
		}
		if ( x != wt - 1 && mask[idx + 1] != 0){
			r += source[(idx + 1) * 3 + i];
		}
		if( y != 0 && mask[(idx - wt)] != 0){
			r += source[(idx - wt) * 3 + i];
		}
		if( y != ht - 1 && mask[(idx + wt)] != 0){
			r += source[(idx + wt) * 3 + i];
		}
		dest[idx * 3 + i] = r/4;
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
	float *fixed, *buf1, * buf2;
	cudaMalloc(&fixed, 3*wt*ht*sizeof(float));
	cudaMalloc(&buf1, 3*wt*ht*sizeof(float));
	cudaMalloc(&buf2, 3*wt*ht*sizeof(float));

	dim3 gdim(CeilDiv(wt, 32), CeilDiv(ht, 16)), bdim(32,16);
	CalculateFixed<<<gdim, bdim>>>(
		background, target, mask, fixed,
		wb, hb, wt, ht, oy, ox
	);
	cudaMemcpy(buf1, target, sizeof(float)*3*wt*ht, cudaMemcpyDeviceToDevice);

	cudaError_t error = cudaDeviceSynchronize();
	if( error != cudaSuccess) cout << cudaGetErrorString(error) << endl;

	for (int i = 0 ; i < 20000; i++){
		PoissonImageCloningIteration<<<gdim, bdim>>>(
			fixed, mask, buf1, buf2, wt, ht
		);
		PoissonImageCloningIteration<<<gdim, bdim>>>(
			fixed, mask, buf2, buf1, wt, ht
		);
	}
	error = cudaDeviceSynchronize();
	if( error != cudaSuccess) cout << cudaGetErrorString(error) << endl;

	cudaMemcpy(output, background, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);
	SimpleClone<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
		background, buf1, mask, output,
		wb, hb, wt, ht, oy, ox
	);
}


