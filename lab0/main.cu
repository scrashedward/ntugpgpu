#include <cstdio>
#include <cstdlib>
#include "SyncedMemory.h"

#define CHECK {\
	auto e = cudaDeviceSynchronize();\
	if (e != cudaSuccess) {\
		printf("At " __FILE__ ":%d, %s\n", __LINE__, cudaGetErrorString(e));\
		abort();\
	}\
}

const int W = 40;
const int H = 12;

__global__ void Draw(char *frame) {
	// TODO: draw more complex things here
	// Do not just submit the original file provided by the TA!
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (y < H and x < W) {
		char c;
		if (x == W-1) {
			c = y == H-1 ? '\0' : '\n';
		} else if (y == 0 or y == H-1 or x == 0 or x == W-2) {
			c = ':';
		} else if (
		(y == 2 && (x == 3 || x == 10 || x == 28 || x == 30)) ||
		(y == 3 && (x == 3 || x == 4 || x == 10 || x == 28 || x == 30)) ||
		(y == 4 && (x == 3 || x == 5 || x == 10 || x == 28 || x == 30)) ||
		(y == 5 && !(x == 1 || x == 2 || x == 4 || x == 5 || x == 7 || x == 8 || x == 9 || x == 11 || x == 17 || x == 23 || x == 29 || x == 35 || x == 36 || x == 37 || x == 38)) ||
		(y == 6 && (x == 3 || x == 7 || x == 10 || x == 28 || x == 30 || x == 12 || x == 16 || x == 20 || x == 24 || x == 34 )) ||
		(y == 7 && (x == 3 || x == 8 || x == 10 || x == 28 || x == 30 || x == 12 || x == 16 || x == 20 || x == 24 || x == 34 || x == 13 || x == 14 || x == 15)) ||
		(y == 8 && (x == 3 || x == 9 || x == 10 || x == 28 || x == 30 || x == 12 || x == 20 || x == 24 || x == 34 )) ||
		(y == 9 && !(x == 1 || x == 2 || x == 6 || x == 4 || x == 5 || x == 7 || x == 8 || x == 9 || x == 11 || x == 17 || x == 23 || x == 29 || x == 35 || x == 36 || x == 37 || x == 38 || x == 18 || x == 19 || x == 21 || x == 22 ))
		){
			c = '#';
		} else{
			c = ' ';
		}
		frame[y*W+x] = c;
	}
}

int main(int argc, char **argv)
{
	MemoryBuffer<char> frame(W*H);
	auto frame_smem = frame.CreateSync(W*H);
	CHECK;

	Draw<<<dim3((W-1)/16+1,(H-1)/12+1), dim3(16,12)>>>(frame_smem.get_gpu_wo());
	CHECK;

	puts(frame_smem.get_cpu_ro());
	CHECK;
	return 0;
}