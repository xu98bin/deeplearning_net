#pragma once
#include "tensor.h"

class IMG2COL {
public:
	static void img2col_cpu(float* input, int in_ch, int h, int w, int kernel_size, int stride, int padding, float* output);
	static void col2img_cpu(float* delta_output, int in_ch, int h, int w, int kernel_size, int stride, int padding, float* delta_input);
};