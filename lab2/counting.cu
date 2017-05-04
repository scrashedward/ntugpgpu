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
__global__ void count1(const char* text, int *pos, int text_size);
__global__ void count2(const char* text, int *pos, int text_size);

int r = 650;

struct increase : thrust::unary_function<int, int>{
	__device__ int operator()(int x){ return x + 1; }
};

struct notendl : thrust::unary_function<char, bool>{
	__device__ int operator()(char x){ return x != '\n'; }
};

void CountPosition1(const char *text, int *pos, int text_size)
{
	increase inc;
	notendl noe;
	thrust::device_ptr<const char> t = thrust::device_pointer_cast(text);
	thrust::device_ptr<int> p = thrust::device_pointer_cast(pos);
	thrust::transform_if(p, p + text_size, t, p, inc, noe);
	thrust::exclusive_scan_by_key(p, p + text_size, p, p);
	thrust::transform_if(p, p + text_size, t, p, inc, noe);
	//int * test = new int[text_size];
	//char * testc = new char[text_size];
	//cudaMemcpy(test, pos, sizeof(int)*text_size, cudaMemcpyDeviceToHost);
	//cudaMemcpy(testc, text, sizeof(char)*text_size, cudaMemcpyDeviceToHost);
	//for(int i = r ; i < r + 50; i++){
	//	cout << test[i] << " ";
	//}
	//cout << endl;
	//for(int i = r ; i < r + 50; i++){
	//	cout << testc[i] << " ";
	//}
	//cout << endl;
	
}

void CountPosition2(const char *text, int *pos, int text_size)
{
	count1<<<(text_size/512 + 1), 512>>>(text, pos, text_size);
}


__global__ void count1(const char* text, int *pos, int text_size){

	__shared__ short s[1024];
	register int tid = threadIdx.x;
	int idx = blockIdx.x * 512 + tid;

	if( idx - 512 >= 0 && text[idx - 512] != '\n'){
		s[tid] = 1;
	}
	else s[tid] = 0;

	if ( idx < text_size && text[idx] != '\n') s[tid + 512] = 1;
	else s[tid + 512] = 0;
	register short t1 = s[tid];
	register short t2 = s[tid+512];
	register short j = 1;
	for (int i = 0; i < 9; ++i){
		__syncthreads();
		if(s[tid + 512] == j){
			t2 += s[tid + 512 - j];
		}
		if(s[tid] == 1<<i && tid - j >= 0){
			t1 += s[tid - j];
		}
		__syncthreads();
		s[tid + 512] = t2;
		s[tid] = t1;
		j<<=1;
	}
	if(idx>text_size) return;
	pos[idx] = s[tid + 512];
}

__global__ void count2(const char* text, int *pos, int text_size){

	__shared__ short s[1024];
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int idx = bid * 512 + tid;

	if( idx - 512 >= 0 && text[idx - 512] != '\n'){
		s[tid] = 1;
	}
	else s[tid] = 0;

	if ( idx < text_size && text[idx] != '\n') s[tid + 512] = 1;
	else s[tid + 512] = 0;
	if(idx>text_size) return;
	__syncthreads();
	int i = 0;
	for(; i < 501; i++){
		if(s[tid+512-i]==0) break;
	}
	pos[idx] = i;
}
