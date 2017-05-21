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

	if(bgx < 0) bgx = -bgx;
	else if( bgx > wb ) bgx = 2 * wb - bgx;

	if(bgy < 0) bgy = -bgy;
	else if (bgy > hb ) bgy = 2 * hb - bgy;

	int bidx = bgy * wb + bgx;

	for(int i=0;i<3;i++){
		if(mask[idx] < 127){
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

		if( x == 0 || mask[idx-1] < 127){
			if(bgx == 0) r += background[bidx * 3 + i];
			else r += background[(bidx - 1) * 3 + i];
		}
		if( x == wt-1 || mask[idx+1] < 127){
			if(bgx == wb-1) r += background[bidx * 3 + i];
			else r += background[(bidx + 1) * 3 + i];
		}
		if( y == 0 || mask[idx-wt] < 127){
			if(bgy == 0 ) r += background[bidx * 3 + i];
			else r += background[(bidx - wb) * 3 + i];
		}
		if( y == ht -1 || mask[idx+wt] < 127){
			if(bgy == hb-1) r += background[bidx * 3 + i];
			else r += background[(bidx + wb) * 3 + i];
		}

		fixed[idx * 3 + i] = r;
	}
}

__global__ void PoissonImageCloningIteration(
	const float *fixed,
	const float *mask,
	float *source, float *dest, int wt, int ht, float w
)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int x = bx * blockDim.x + tx;
	int y = by * blockDim.y + ty;
	int idx = y * wt + x;

	for(int i = 0; i < 3; i++){
		if (mask[idx] < 127 ){
			dest[idx * 3 + i] = fixed[idx * 3 + i];
			continue;
		}
		float r = fixed[idx * 3 + i];
		if ( x != 0 && mask[idx-1] > 127 ){
			r += source[(idx - 1)* 3+ i];
		}
		if ( x != wt - 1 && mask[idx + 1] > 127){
			r += source[(idx + 1) * 3 + i];
		}
		if( y != 0 && mask[(idx - wt)] > 127){
			r += source[(idx - wt) * 3 + i];
		}
		if( y != ht - 1 && mask[(idx + wt)] > 127){
			r += source[(idx + wt) * 3 + i];
		}
		dest[idx * 3 + i] = r/4 * w + (1-w) * source[idx * 3 + i];
	}
}

__global__ void DownSample(
	const float *input,
	float *output,
	int wt, int ht
)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int x = bx * 32 + tx * 2;
	int y = by * 16 + ty * 2;

	for(int i = 0; i < 3; i++){
		float r = 0;
		r += input[(y*wt + x) * 3 + i];
		r += input[((y+1)*wt + x) * 3 + i];
		r += input[(y * wt + x + 1) * 3 + i];
		r += input[((y+1)*wt + x + 1) * 3 + i];
		output[(y/2*wt/2 + x / 2)*3 + i] = r/4;
	}

}

__global__ void UpSample(
	const float *input,
	float *output,
	int wt, int ht
)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;

	int x = bx * 16 + tx;
	int y = by * 8 + ty;
	int idx = y * wt + x;

	float lu = 0, ld = 0, ru = 0, rd = 0;

	for(int i = 0 ; i < 3 ; i++ ){
		if(x == 0){
			if(y == 0){
				lu = input[idx * 3 + i];
				ld = (input[idx * 3 + i] * 4 + input[(idx + wt) * 3 + i] * 2)/6;
				ru = (input[idx * 3 + i] * 4 + input[(idx + 1) * 3 + i] * 2)/6;
				rd = (input[idx * 3 + i] * 4 + input[(idx + 1) * 3 + i] * 2 + input[(idx + wt) * 3 + i] * 2 + input[(idx + wt + 1) * 3 + i] * 1) / 9;
			}
			else if (y == ht-1) {
				lu = (input[idx * 3 + i] * 4 + input[(idx - wt) * 3 + i] * 2)/6;
				ld = input[idx * 3 + i];
				rd = (input[idx * 3 + i] * 4 + input[(idx + 1) * 3 + i] * 2)/6;
				ru = (input[idx * 3 + i] * 4 + input[(idx + 1) * 3 + i] * 2 + input[(idx - wt) * 3 + i] * 2 + input[(idx - wt + 1) * 3 + i] * 1) / 9;
			}
			else{
				lu = (input[idx * 3 + i] * 4 + input[(idx - wt) * 3 + i] * 2)/6;
				ld = (input[idx * 3 + i] * 4 + input[(idx + wt) * 3 + i] * 2)/6;
				ru = (input[idx * 3 + i] * 4 + input[(idx + 1) * 3 + i] * 2 + input[(idx - wt) * 3 + i] * 2 + input[(idx - wt + 1) * 3 + i] * 1) / 9;
				rd = (input[idx * 3 + i] * 4 + input[(idx + 1) * 3 + i] * 2 + input[(idx + wt) * 3 + i] * 2 + input[(idx + wt + 1) * 3 + i] * 1) / 9;
			}
		}else if(x == wt -1){
			if(y == 0){
				ru = input[idx * 3 + i];
				rd = (input[idx * 3 + i] * 4 + input[(idx + wt) * 3 + i] * 2)/6;
				lu = (input[idx * 3 + i] * 4 + input[(idx - 1) * 3 + i] * 2)/6;
				ld = (input[idx * 3 + i] * 4 + input[(idx - 1) * 3 + i] * 2 + input[(idx + wt) * 3 + i] * 2 + input[(idx + wt - 1) * 3 + i] * 1) / 9;
			}
			else if(y == ht-1){
				ru = (input[idx * 3 + i] * 4 + input[(idx - wt) * 3 + i] * 2)/6;
				rd = input[idx * 3 + i];
				ld = (input[idx * 3 + i] * 4 + input[(idx - 1) * 3 + i] * 2)/6;
				lu = (input[idx * 3 + i] * 4 + input[(idx - 1) * 3 + i] * 2 + input[(idx - wt) * 3 + i] * 2 + input[(idx - wt - 1) * 3 + i] * 1) / 9;
			}
			else{
				ru = (input[idx * 3 + i] * 4 + input[(idx - wt) * 3 + i] * 2)/6;
				rd = (input[idx * 3 + i] * 4 + input[(idx + wt) * 3 + i] * 2)/6;
				lu = (input[idx * 3 + i] * 4 + input[(idx - 1) * 3 + i] * 2 + input[(idx - wt) * 3 + i] * 2 + input[(idx - wt - 1) * 3 + i]) / 9;
				ld = (input[idx * 3 + i] * 4 + input[(idx - 1) * 3 + i] * 2 + input[(idx + wt) * 3 + i] * 2 + input[(idx + wt - 1) * 3 + i]) / 9;
			}
		}else if(y == 0){
			ru = (input[idx * 3 + i] * 4 + input[(idx + 1) * 3 + i] * 2) / 6;
			rd = (input[idx * 3 + i] * 4 + input[(idx + 1) * 3 + i] * 2 + input[(idx + wt) * 3 + i] * 2 + input[(idx + wt + 1) * 3 + i]) / 9;
			lu = (input[idx * 3 + i] * 4 + input[(idx - 1) * 3 + i] * 2) / 6;
			ld = (input[idx * 3 + i] * 4 + input[(idx - 1) * 3 + i] * 2 + input[(idx + wt) * 3 + i] * 2 + input[(idx + wt - 1) * 3 + i]) / 9;
		}else if(y == ht-1){
			rd = (input[idx * 3 + i] * 4 + input[(idx + 1) * 3 + i] * 2) / 6;
			ru = (input[idx * 3 + i] * 4 + input[(idx + 1) * 3 + i] * 2 + input[(idx - wt) * 3 + i] * 2 + input[(idx - wt + 1) * 3 + i]) / 9;
			ld = (input[idx * 3 + i] * 4 + input[(idx - 1) * 3 + i] * 2) / 6;
			lu = (input[idx * 3 + i] * 4 + input[(idx - 1) * 3 + i] * 2 + input[(idx - wt) * 3 + i] * 2 + input[(idx - wt - 1) * 3 + i]) / 9;
		}
		else{
			lu = (input[idx * 3 + i] * 4 + input[(idx - 1) * 3 + i] * 2 + input[(idx - wt) * 3 + i] * 2 + input[(idx - wt - 1) * 3 + i]) / 9;
			ld = (input[idx * 3 + i] * 4 + input[(idx - 1) * 3 + i] * 2 + input[(idx + wt) * 3 + i] * 2 + input[(idx + wt - 1) * 3 + i]) / 9;
			ru = (input[idx * 3 + i] * 4 + input[(idx + 1) * 3 + i] * 2 + input[(idx - wt) * 3 + i] * 2 + input[(idx - wt + 1) * 3 + i]) / 9;
			rd = (input[idx * 3 + i] * 4 + input[(idx + 1) * 3 + i] * 2 + input[(idx + wt) * 3 + i] * 2 + input[(idx + wt + 1) * 3 + i]) / 9;
		}
		output[(y * 2 * wt * 2 + x * 2) * 3 + i] = lu;
		output[(y * 2 * wt * 2 + x * 2 + 1) * 3 + i] = ru;
		output[(y * 2 * wt * 2 + wt * 2 + x * 2 + 1) * 3 + i] = rd;
		output[(y * 2 * wt * 2 + wt * 2 + x * 2) * 3 + i] = ld;
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
	float *fixed, *buf1, * buf2, *df, *db1, *db2, *dmask;
	cudaMalloc(&fixed, 3*wt*ht*sizeof(float));
	cudaMalloc(&buf1, 3*wt*ht*sizeof(float));
	cudaMalloc(&buf2, 3*wt*ht*sizeof(float));
	cudaMalloc(&df, 3 * wt * ht * sizeof(float)/4);
	cudaMalloc(&db1, 3 * wt * ht * sizeof(float) / 4);
	cudaMalloc(&db2, 3 * wt * ht * sizeof(float) / 4);
	cudaMalloc(&dmask, 3 * wt * ht * sizeof(float) /4);

	dim3 gdim(CeilDiv(wt, 32), CeilDiv(ht, 16)), bdim(32,16);
	CalculateFixed<<<gdim, bdim>>>(
		background, target, mask, fixed,
		wb, hb, wt, ht, oy, ox
	);
	
	//scaling part, please enable this part for scaling acceleration
	DownSample<<<gdim, dim3(16, 8)>>>(fixed, df, wt, ht);
	DownSample<<<gdim, dim3(16, 8)>>>(target, db1, wt, ht);
	DownSample<<<gdim, dim3(16, 8)>>>(mask, dmask, wt, ht);
	for(int i = 0 ; i < 2500; i++){
		PoissonImageCloningIteration<<<gdim, dim3(16, 8)>>>(
			df, dmask, db1, db2, wt/2, ht/2, 1
		);
		PoissonImageCloningIteration<<<gdim, dim3(16, 8)>>>(
			df, dmask, db2, db1, wt/2, ht/2, 1
		);
	}
	UpSample<<<gdim, dim3(16, 8)>>>(db1, buf1, wt/2, ht/2);
	
	
	//Background acceleration part, please enable this part for background acceleration
	//for(int i = 0; i < ht; i++){
	// 	cudaMemcpy(buf1+i*wt*3, background + (oy*wb+ox) * 3 + i * wb * 3 , sizeof(float)*3*wt, cudaMemcpyDeviceToDevice);
	//}

	// Normal Copy, please disable this part for background acceleration and scaling acceleration
	//cudaMemcpy(buf1, target, sizeof(float)*3*wt*ht, cudaMemcpyDeviceToDevice);

	cudaError_t error = cudaDeviceSynchronize();
	if( error != cudaSuccess) cout << cudaGetErrorString(error) << endl;
	
	for (int i = 0 ; i < 2500; i++){
		float w1 = 1, w2 = 1;
		PoissonImageCloningIteration<<<gdim, bdim>>>(
			fixed, mask, buf1, buf2, wt, ht, w1
		);
		PoissonImageCloningIteration<<<gdim, bdim>>>(
			fixed, mask, buf2, buf1, wt, ht, w2
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


