//module kernel;
#include "kernel.h"

import std;

using namespace std;
using namespace exec;
using namespace exec::cpu;

// padding = ceil( ((x_h - 1) * stride - x_h + k_h) / 2 ) ---> same padding

void conv2d(float* x, float* z, float* kernel, int x_h, int x_w, int c_in, int c_out, int k_h, int k_w, int stride) {
	int z_h = (x_h - k_h) / stride + 1, z_w = (x_w - k_w) / stride + 1;
	int z_idx;

	for (int out_channel = 0; out_channel < c_out; out_channel++) {
		for (int z_row = 0, x_row_start = 0; z_row < z_h; z_row++, x_row_start += stride) {
			if (x_h - x_row_start < k_h) break;
			for (int z_col = 0, x_col_start = 0; z_col < z_w; z_col++, x_col_start += stride) {
				if (x_w - x_col_start < k_w) break;
				z_idx = out_channel * z_h * z_w + z_row * z_w + z_col;
				dot_product_3d(x, kernel, x_h, x_w, c_in, out_channel, k_h, k_w, x_row_start, x_col_start, z + z_idx, nullptr);
			}
		}
	}
}


