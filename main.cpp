#include "kernel.h"
//import kernel;

import std;
using namespace std;

int main() {
	int kernel[] {1, 1, 1, 0, 0, 0, -1, -1, -1};
	int image[]{ 2, 3, 4, 2, 3, 4, 9, 5, 6, 4, 5, 6, 4, 5, 1, 0 };
	int x_h = 4, x_w = 4, k_h = 3, k_w = 3, stride = 1, c_in = 1, c_out = 1;
	int z[2*2];
	// conv2d(image, z, kernel, x_h, x_w, c_in, c_out, k_h, k_w, stride);
	// do_it(image, z, kernel, x_h, x_w, c_in, c_out, k_h, k_w, stride);
	for (int a : z) {
		cout << a << " ";
	}
}
