#include "kernel.h"

__device__ void dot_product_3d(int* x, int* kernel, int x_h, int x_w, int c_in, int out_channel, int k_h, int k_w, int x_row_start, int x_col_start, int *result) {
	int dot_product = 0;
	for (int in_channel = 0; in_channel < c_in; in_channel++) {
		for (int k_row = 0, x_row = x_row_start; k_row < k_h; k_row++, x_row++) {
			for (int k_col = 0, x_col = x_col_start; k_col < k_w; k_col++, x_col++) {
				int x_idx = in_channel * x_h * x_w + x_row * x_w + x_col;
				int k_idx = out_channel * c_in * k_h * k_w + in_channel * k_h * k_w + k_row * k_w + k_col;
				dot_product += x[x_idx] * kernel[k_idx];
			}
		}
	}
	*result = dot_product;
	return;
}

__global__ void conv2d_gpu (int* x, int* z, int* kernel, int x_h, int x_w, int c_in, int c_out, int k_h, int k_w, int stride) {
	int z_h = (x_h - k_h) / stride + 1, z_w = (x_w - k_w) / stride + 1;
	int z_idx;

	int out_channel = gridDim.x * blockDim.x;
	int z_row = gridDim.y * blockDim.y;
	int z_col = gridDim.z * blockDim.z;

	int x_row_start = z_row * stride, x_col_start = z_col * stride;

	if (out_channel >= c_out || x_h - x_row_start < k_h || x_w - x_col_start < k_w) return;

	z_idx = out_channel * z_h * z_w + z_row * z_w + z_col;
	dot_product_3d(x, kernel, x_h, x_w, c_in, out_channel, k_h, k_w, x_row_start, x_col_start, z + z_idx);
}

void do_it(int* x, int* z, int* kernel, int x_h, int x_w, int c_in, int c_out, int k_h, int k_w, int stride) {
	int z_h = (x_h - k_h) / stride + 1, z_w = (x_w - k_w) / stride + 1;
	int* d_x, * d_z, * d_kernel;

	int x_size = c_in * x_h * x_w * sizeof(int), z_size = c_out * z_h * z_w * sizeof(int), kernel_size = c_out * c_in * k_h * k_w * sizeof(int);

	cudaMalloc(&d_x, x_size);
	cudaMalloc(&d_z, z_size);
	cudaMalloc(&d_kernel, kernel_size);

	cudaMemcpy(d_x, x, x_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_z, z, z_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_kernel, kernel, kernel_size, cudaMemcpyHostToDevice);

	unsigned int threads_h = 1024 * z_h / (z_h + z_w + c_out), threads_w = 1024 * z_w / (z_h + z_w + c_out);
	unsigned int threads_c = 1024 - threads_h - threads_w;

	conv2d_gpu << < {(unsigned int) ceil(c_out / threads_c), (unsigned int) ceil(z_h / threads_h), (unsigned int) ceil(z_w / threads_w)},
	{ threads_c, threads_h, threads_w } >> > (d_x, d_z, d_kernel, x_h, x_w, c_in, c_out, k_h, k_w, stride);

	cudaMemcpy(z, d_z, z_size, cudaMemcpyDeviceToHost);

	cudaFree(d_x);
	cudaFree(d_z);
	cudaFree(d_kernel);

}
