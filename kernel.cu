#include "kernel.h"
#include <cuda/std/numeric>

using namespace exec;
using namespace exec::gpu;

__host__ __device__ float exec::relu(float input) {
	return (input >= 0) ? input : -input;
}

__host__ __device__ float exec::sigmoid(float input) {
	return 1 / (1 + expf(-input));
}

__host__ __device__ float exec::tanh(float input) {
	return (expf(input) - expf(-input)) / (expf(input) + expf(-input));
}

__host__ __device__ void exec::dot_product_3d(float* x, float* kernel, int x_h, int x_w, int c_in, int out_channel, int k_h, int k_w, int x_row_start, int x_col_start, float* result, float (*act_fn) (float)) {
	float dot_product = 0.f;
	for (int in_channel = 0; in_channel < c_in; in_channel++) {
		for (int k_row = 0, x_row = x_row_start; k_row < k_h; k_row++, x_row++) {
			for (int k_col = 0, x_col = x_col_start; k_col < k_w; k_col++, x_col++) {
				int x_idx = in_channel * x_h * x_w + x_row * x_w + x_col;
				int k_idx = out_channel * c_in * k_h * k_w + in_channel * k_h * k_w + k_row * k_w + k_col;
				dot_product += x[x_idx] * kernel[k_idx];
			}
		}
	}
	*result = act_fn ? act_fn(dot_product) : dot_product;
	return;
}

__host__ __device__ void exec::pool2d_kernel(float* x, int x_h, int x_w, int out_channel, int k_h, int k_w, int x_row_start, int x_col_start, float* result, int type, float(*act_fn) (float)) {
	float pool_value;
	if (type == 0) pool_value = ::cuda::std::numeric_limits<float>::min();
	else if (type == 1) pool_value = 0.f;

	for (int k_row = 0, x_row = x_row_start; k_row < k_h; k_row++, x_row++) {
		for (int k_col = 0, x_col = x_col_start; k_col < k_w; k_col++, x_col++) {
			int x_idx = out_channel * x_h * x_w + x_row * x_w + x_col;

			if (type == 0) pool_value = (x[x_idx] > pool_value) ? x[x_idx] : pool_value;
			else if (type == 1) pool_value += x[x_idx];
		}
	}
	if (type == 1) pool_value /= k_h * k_w;
	*result = act_fn ? act_fn(pool_value) : pool_value;
	return;
}

__global__ void conv2d(float* x, float* z, float* kernel, int x_h, int x_w, int c_in, int c_out, int k_h, int k_w, int stride, float(*act_fn) (float)) {
	int z_h = (x_h - k_h) / stride + 1, z_w = (x_w - k_w) / stride + 1;
	int z_idx;

	int out_channel = blockIdx.x * blockDim.x + threadIdx.x;
	int z_row = blockIdx.y * blockDim.y + threadIdx.y;
	int z_col = blockIdx.z * blockDim.z + threadIdx.z;

	int x_row_start = z_row * stride, x_col_start = z_col * stride;

	if (out_channel >= c_out || x_h - x_row_start < k_h || x_w - x_col_start < k_w) return;

	z_idx = out_channel * z_h * z_w + z_row * z_w + z_col;
	dot_product_3d(x, kernel, x_h, x_w, c_in, out_channel, k_h, k_w, x_row_start, x_col_start, z + z_idx, act_fn);
}

__device__ void pooling2d(float* x, float* z, int x_h, int x_w, int c_in, int k_h, int k_w, int stride_h, int stride_w, int type, float(*act_fn) (float)) {
	int z_h = (x_h - k_h) / stride_h + 1, z_w = (x_w - k_w) / stride_w + 1;
	int z_idx;

	int out_channel = blockIdx.x * blockDim.x + threadIdx.x;
	int z_row = blockIdx.y * blockDim.y + threadIdx.y;
	int z_col = blockIdx.z * blockDim.z + threadIdx.z;

	int x_row_start = z_row * stride_h, x_col_start = z_col * stride_w;

	if (out_channel >= c_in || x_h - x_row_start < k_h || x_w - x_col_start < k_w) return;

	z_idx = out_channel * z_h * z_w + z_row * z_w + z_col;
	pool2d_kernel(x, x_h, x_w, out_channel, k_h, k_w, x_row_start, x_col_start, z + z_idx, 0, act_fn);
}

__global__ void maxPooling2d(float* x, float* z, int x_h, int x_w, int c_in, int k_h, int k_w, int stride_h, int stride_w, float(*act_fn) (float)) {
	pooling2d(x, z, x_h, x_w, c_in, k_h, k_w, stride_h, stride_w, 0, act_fn);
}

__global__ void avgPooling2d(float* x, float* z, int x_h, int x_w, int c_in, int k_h, int k_w, int stride_h, int stride_w, float(*act_fn) (float)) {
	pooling2d(x, z, x_h, x_w, c_in, k_h, k_w, stride_h, stride_w, 1, act_fn);
}

float* exec::gpu::conv2d_layer(float* x, float* kernel, int x_h, int x_w, int c_in, int c_out, int k_h, int k_w, int stride, float(*act_fn) (float)) {
	int z_h = (x_h - k_h) / stride + 1, z_w = (x_w - k_w) / stride + 1;
	float* d_x, * d_z, * d_kernel;

	int x_size = c_in * x_h * x_w * sizeof(float), z_size = c_out * z_h * z_w * sizeof(float), kernel_size = c_out * c_in * k_h * k_w * sizeof(float);

	float* z = (float*) malloc(z_size);

	cudaMalloc(&d_x, x_size);
	cudaMalloc(&d_z, z_size);
	cudaMalloc(&d_kernel, kernel_size);

	cudaMemcpy(d_x, x, x_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_z, z, z_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_kernel, kernel, kernel_size, cudaMemcpyHostToDevice);

	unsigned int threads_h = 1024 * z_h / (z_h + z_w + c_out), threads_w = 1024 * z_w / (z_h + z_w + c_out);
	unsigned int threads_c = 1024 - threads_h - threads_w;

	conv2d<<< {(unsigned int)ceil(c_out / threads_c), (unsigned int)ceil(z_h / threads_h), (unsigned int)ceil(z_w / threads_w)},
	{ threads_c, threads_h, threads_w } >>>(d_x, d_z, d_kernel, x_h, x_w, c_in, c_out, k_h, k_w, stride, act_fn);

	cudaMemcpy(z, d_z, z_size, cudaMemcpyDeviceToHost);

	cudaFree(d_x);
	cudaFree(d_z);
	cudaFree(d_kernel);

	return z;
}

float* exec::gpu::maxpool_layer(float* x, int x_h, int x_w, int c_in, int k_h, int k_w, int stride_h, int stride_w, float(*act_fn) (float)) {
	int z_h = (x_h - k_h) / stride_h + 1, z_w = (x_w - k_w) / stride_w + 1;
	float* d_x, * d_z;
	int c_out = c_in;

	int x_size = c_in * x_h * x_w * sizeof(float), z_size = c_out * z_h * z_w * sizeof(float);

	float* z = (float*)malloc(z_size);

	cudaMalloc(&d_x, x_size);
	cudaMalloc(&d_z, z_size);

	cudaMemcpy(d_x, x, x_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_z, z, z_size, cudaMemcpyHostToDevice);

	unsigned int threads_h = 1024 * z_h / (z_h + z_w + c_out), threads_w = 1024 * z_w / (z_h + z_w + c_out);
	unsigned int threads_c = 1024 - threads_h - threads_w;

	maxPooling2d << < {(unsigned int)ceil(c_out / threads_c), (unsigned int)ceil(z_h / threads_h), (unsigned int)ceil(z_w / threads_w)},
	{ threads_c, threads_h, threads_w } >> > (d_x, d_z, x_h, x_w, c_in, k_h, k_w, stride_h, stride_w, act_fn);

	cudaMemcpy(z, d_z, z_size, cudaMemcpyDeviceToHost);

	cudaFree(d_x);
	cudaFree(d_z);

	return z;
}

float* exec::gpu::avgpool_layer(float* x, int x_h, int x_w, int c_in, int k_h, int k_w, int stride_h, int stride_w, float(*act_fn) (float)) {
	int z_h = (x_h - k_h) / stride_h + 1, z_w = (x_w - k_w) / stride_w + 1;
	float* d_x, * d_z;
	int c_out = c_in;

	int x_size = c_in * x_h * x_w * sizeof(float), z_size = c_out * z_h * z_w * sizeof(float);

	float* z = (float*)malloc(z_size);

	cudaMalloc(&d_x, x_size);
	cudaMalloc(&d_z, z_size);

	cudaMemcpy(d_x, x, x_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_z, z, z_size, cudaMemcpyHostToDevice);

	unsigned int threads_h = 1024 * z_h / (z_h + z_w + c_out), threads_w = 1024 * z_w / (z_h + z_w + c_out);
	unsigned int threads_c = 1024 - threads_h - threads_w;

	avgPooling2d << < {(unsigned int)ceil(c_out / threads_c), (unsigned int)ceil(z_h / threads_h), (unsigned int)ceil(z_w / threads_w)},
	{ threads_c, threads_h, threads_w } >> > (d_x, d_z, x_h, x_w, c_in, k_h, k_w, stride_h, stride_w, act_fn);

	cudaMemcpy(z, d_z, z_size, cudaMemcpyDeviceToHost);

	cudaFree(d_x);
	cudaFree(d_z);

	return z;
}

void do_it(float* x, float* z, float* kernel, int x_h, int x_w, int c_in, int c_out, int k_h, int k_w, int stride) {
	int z_h = (x_h - k_h) / stride + 1, z_w = (x_w - k_w) / stride + 1;
	float* d_x, * d_z, * d_kernel;

	int x_size = c_in * x_h * x_w * sizeof(float), z_size = c_out * z_h * z_w * sizeof(float), kernel_size = c_out * c_in * k_h * k_w * sizeof(float);

	cudaMalloc(&d_x, x_size);
	cudaMalloc(&d_z, z_size);
	cudaMalloc(&d_kernel, kernel_size);

	cudaMemcpy(d_x, x, x_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_z, z, z_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_kernel, kernel, kernel_size, cudaMemcpyHostToDevice);

	unsigned int threads_h = 1024 * z_h / (z_h + z_w + c_out), threads_w = 1024 * z_w / (z_h + z_w + c_out);
	unsigned int threads_c = 1024 - threads_h - threads_w;

	conv2d << < {(unsigned int) ceil(c_out / threads_c), (unsigned int) ceil(z_h / threads_h), (unsigned int) ceil(z_w / threads_w)},
	{ threads_c, threads_h, threads_w } >> > (d_x, d_z, d_kernel, x_h, x_w, c_in, c_out, k_h, k_w, stride, nullptr);

	cudaMemcpy(z, d_z, z_size, cudaMemcpyDeviceToHost);

	cudaFree(d_x);
	cudaFree(d_z);
	cudaFree(d_kernel);
}
