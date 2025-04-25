#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <stdlib.h>

//#ifdef __CUDA_ARCH__
//#define AA gpu
//#else
//#define AA cpu
//#endif

void do_it(float* x, float* z, float* kernel, int x_h, int x_w, int c_in, int c_out, int k_h, int k_w, int stride);

namespace exec {
	namespace cpu {

	}

	namespace gpu {
		float* conv2d_layer(float* x, float* kernel, int x_h, int x_w, int c_in, int c_out, int k_h, int k_w, int stride, float(*act_fn) (float) = nullptr);
		float* maxpool_layer(float* x, int x_h, int x_w, int c_in, int k_h, int k_w, int stride_h, int stride_w, float(*act_fn) (float) = nullptr);
		float* avgpool_layer(float* x, int x_h, int x_w, int c_in, int k_h, int k_w, int stride_h, int stride_w, float(*act_fn) (float) = nullptr);
	}

	__host__ __device__ void dot_product_3d(float* x, float* kernel, int x_h, int x_w, int c_in, int out_channel, int k_h, int k_w, int x_row_start, int x_col_start, float* result, float (*act_fn) (float) = nullptr);
	__host__ __device__ void pool2d_kernel(float* x, int x_h, int x_w, int out_channel, int k_h, int k_w, int x_row_start, int x_col_start, float* result, int type, float(*act_fn) (float) = nullptr);

	__host__ __device__ float relu(float input);
	__host__ __device__ float sigmoid(float input);
	__host__ __device__ float tanh(float input);
}
