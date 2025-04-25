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

void do_it(int* x, int* z, int* kernel, int x_h, int x_w, int c_in, int c_out, int k_h, int k_w, int stride);

namespace exec {
	namespace cpu {

	}

	namespace gpu {

	}
}
