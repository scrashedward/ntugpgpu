#include "lab1.h"
static const unsigned W = 640;
static const unsigned H = 480;
static const unsigned NFRAME = 480;
static const unsigned s[9] = {0,0,0,1,1,1,2,2,2};
static const unsigned slen = 9; 

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
	float ra[3] = { 1.8 + (1.8/(float)640)*float(x), 1.8 + (1.8/(float)480)*float(y), 2.8 + (0.5/float(480))*float(t)};
	float r1;
	float x1= 0.5,s=0;
	for(int i = 0 ; i < 150;i++){
		r1 = ra[gs[i%9]];
		x1 = r1*x1*(1-x1);
		s += logf(0.00001 + fabs(r1*(1-2*x1)));
	}
	s/=150;
	s*=205;
	s+=50;
	//s+=100;
	float r,g,b;
	r = 175-s;
	g = (s-70)*1.5;
	if(r<0)r=0;
	if(r>255) r=255;
	if(g>255) g=255;
	b = 100-s;
	if(g<0) g=0;
	if(b<0) b=0;
	yuv[y*640+x] = (uint8_t)(r*0.299+g*0.587+b*0.114);
	if(!y%2&&!x%2)
	yuv[640*480+(320*(y/2)+x/2)] = -0.169*r-0.331*g + b*0.5+128;
	yuv[640*600+(320*(y/2)+x/2)] = 0.5*r-0.419*g-0.081*b + 128;
}
