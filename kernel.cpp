module kernel;

import std;

using namespace std;

int dot_product_3d(int* x, int* kernel, int x_h, int x_w, int c_in, int out_channel, int k_h, int k_w, int x_row_start, int x_col_start) {
	int result = 0;
	for (int in_channel = 0; in_channel < c_in; in_channel++) {
		for (int k_row = 0, x_row = x_row_start; k_row < k_h; k_row++, x_row++) {
			for (int k_col = 0, x_col = x_col_start; k_col < k_w; k_col++, x_col++) {
				int x_idx = in_channel * x_h * x_w + x_row * x_w + x_col;
				int k_idx = out_channel * c_in * k_h * k_w + in_channel * k_h * k_w + k_row * k_w + k_col;
				result += x[x_idx] * kernel[k_idx];
			}
		}
	}
	return result;
}

// padding = ceil( ((x_h - 1) * stride - x_h + k_h) / 2 ) ---> same padding

void conv2d(int* x, int* z, int* kernel, int x_h, int x_w, int c_in, int c_out, int k_h, int k_w, int stride) {
	int z_h = (x_h - k_h) / stride + 1, z_w = (x_w - k_w) / stride + 1;
	int x_idx, k_idx, z_idx, z_ele;

	for (int out_channel = 0; out_channel < c_out; out_channel++) {
		for (int z_row = 0, x_row_start = 0; z_row < z_h; z_row++, x_row_start += stride) {
			if (x_h - x_row_start < k_h) break;
			for (int z_col = 0, x_col_start = 0; z_col < z_w; z_col++, x_col_start += stride) {
				if (x_w - x_col_start < k_w) break;
				z_idx = out_channel * z_h * z_w + z_row * z_w + z_col;
				z[z_idx] = dot_product_3d(x, kernel, x_h, x_w, c_in, out_channel, k_h, k_w, x_row_start, x_col_start);
			}
		}
	}
}


