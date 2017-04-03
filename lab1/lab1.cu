#include "lab1.h"
#include "math_constants.h"
static const unsigned W = 640;
static const unsigned H = 480;
static const unsigned NFRAME = 960;
static const unsigned s[13] = {1,1,1,1,1,1,2,0,0,0,0,0,0};
static const unsigned slen = 13;

__global__ void g(uint8_t *yuv, int* gs, int t);
__global__ void debug(uint8_t *yuv);


struct Lab1VideoGenerator::Impl {
	int t = 0;
	int *gs = 0;
};

Lab1VideoGenerator::Lab1VideoGenerator(): impl(new Impl) {
	cudaMalloc(&impl->gs, sizeof(int)*slen);
	cudaMemcpy(impl->gs, s, sizeof(int)*slen , cudaMemcpyHostToDevice);
}

Lab1VideoGenerator::~Lab1VideoGenerator() {}

void Lab1VideoGenerator::get_info(Lab1VideoInfo &info) {
	info.w = W;
	info.h = H;
	info.n_frame = NFRAME;
	// fps = 24/1 = 24
	info.fps_n = 24;
	info.fps_d = 1;
};


void Lab1VideoGenerator::Generate(uint8_t *yuv) {
	//cudaMemset(yuv, (impl->t)*255/NFRAME, W*H);
	cudaMemset(yuv+W*H, 0, W*H/2);
	cudaDeviceSynchronize();
	g<<<H,W>>>(yuv, impl->gs, impl->t);
	cudaDeviceSynchronize();
	++(impl->t);
	//debug<<<1,1>>>(yuv);
}

__global__ void g(uint8_t *yuv, int* gs, int t){
	int x = threadIdx.x;
	int y = blockIdx.x;
	float ra[3] = { 4 - (0.6/(float)640)*float(y), 2.5 + (0.8/(float)480)*float(x), 0 + (4/float(960))*float(t)};
	float r1;
	float x1= 0.5,s=0;
	for(int i = 0 ; i < 200;i++){
		r1 = ra[gs[i%13]];
		x1 = r1*x1*(1-x1);
		s += logf(0.00001 + fabs(r1*(1-2*x1)));
	}
	s = (atanf(s) + CUDART_PI_F/2)/CUDART_PI_F;
	float r,g,b;
	r = (1-s)*255;
	g = (1-s)*193;
	b = (1-s)*37+s*205;
	yuv[y*640+x] = (uint8_t)(r*0.299 + g*0.587 + b*0.114);
	if(y%2==0 && x%2==0){
		yuv[640*480+(320*(y/2)+x/2)] = -0.169*r - g*0.289 + 0.5*b + 128;
		yuv[640*600+(320*(y/2)+x/2)] = 0.5*r - g*0.515 - 0.081*b + 128;
	}

}
