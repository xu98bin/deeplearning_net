#pragma once
#include "model.h"
#include "img2col.h"
#include "tensor.h"

class ConvLayer :public Module {
private:
	void add_bias(Tensor& output);
	float sum_array(float* data, size_t size);
	void backward_bias(Tensor& delta);
	void sum_gradient_weight(Tensor&& batch_gradient_weight);
public:
	int kernel_size, stride, padding, in_channels, out_channels, out_width, out_height;
	Tensor weights, bias_weight, gradient_weight, gradient_bias, gradient_input, col_input;
	bool use_bias;
	vector<int> input_shape, output_shape;
public:
	ConvLayer(int k, int s, int p, int in_ch, int out_ch, bool bias);
	Tensor forward(Tensor& input);
	pair<Tensor, bool> backward(Tensor& delta, bool last_required_grad);
	void update_paramers(const float lr);
};