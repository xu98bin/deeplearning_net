#include "convlayer.h"

ConvLayer::ConvLayer(int k, int s, int p, int in_ch, int out_ch, bool bias) :kernel_size(k), stride(s), padding(p), in_channels(in_ch), out_channels(out_ch), weights(vector<int>{out_ch, in_ch* k* k}),use_bias(bias) {
	if (bias) bias_weight = Tensor(vector<int>{out_ch});
	weights.normalTensor(0.0f, 0.02f);
}

Tensor ConvLayer::forward(Tensor& input) {
	//input.shape=[N,IC,IH,IW]
	//output.shape[N,OC,OH,OW]
	assert(input.dims() == 4);
	input_shape = input.shape;
	vector<int> col_shape(3, 0);
	col_shape[0] = input.size(0);
	col_shape[1] = weights.size(1);
	out_width = (input.size(3) + 2 * padding - kernel_size) / stride + 1;
	out_height = (input.size(2) + 2 * padding - kernel_size) / stride + 1;
	col_shape[2] = out_width * out_height;
	Tensor img2col_output(col_shape);
	col_input = std::move(img2col_output);
	int input_img_size = input.size(1) * input.size(2) * input.size(3);
	int col_img_size = col_shape[1] * col_shape[2];
	for (int i = 0; i < col_shape[0]; i++) {
		float* img = input.data + i * input_img_size;
		float* col = col_input.data + i * col_img_size;
		IMG2COL::img2col_cpu(img, input.size(1), input.size(2), input.size(3),
			kernel_size, stride, padding, col);
	}
	Tensor output = matmul(weights, col_input);
	assert(output.dims() == 3);
	add_bias(output);
	output.shape[2] = out_height, output.shape.push_back(out_width);
	output_shape = output.shape;
	return std::move(output);
}

void ConvLayer::add_bias(Tensor& output) {
	if (bias_weight.numel() > 0) {
		assert(bias_weight.numel() == out_channels);
		int batch_size = output.size(1) * output.size(2);
		for (int b = 0; b < output.size(0); b++) {
			int batch_pos = b * batch_size;
			for (int c = 0; c < out_channels; c++) {
				int channel_pos = c * output.size(2);
				for (int i = 0; i < output.size(2); i++)
					output.data[batch_pos + channel_pos + i] += bias_weight.data[c];
			}
		}
	}
}

void ConvLayer::sum_gradient_weight(Tensor&& batch_gradient_weight) {
	gradient_weight = Tensor(weights.shape);
	int last_dim = weights.numel();
	int batches = batch_gradient_weight.numel()/last_dim;
	int b;
	for (b = 0; b < batches; b++) {
		float* sing_grad = &batch_gradient_weight.data[b * last_dim];
		for (int i = 0; i < last_dim; i++)
			gradient_weight.data[i] += sing_grad[i];
	}
}

float ConvLayer::sum_array(float* data, size_t size) {
	float ans = 0;
	for (int i = 0; i < size; i++)
		ans += data[i];
	return ans;
}

void ConvLayer::backward_bias(Tensor& delta) {
	assert(delta.size(1) == out_channels);
	int batch_size = out_channels * out_width * out_height;
	int channel_size = out_width * out_height;
	int b;
	for (b = 0; b < delta.size(0); b++) {
		for (int c = 0; c < out_channels; c++) {
			gradient_bias.data[c] += sum_array(&delta.data[b * batch_size + c * channel_size], channel_size);
		}
	}
}

pair<Tensor, bool> ConvLayer::backward(Tensor& delta, bool last_required_grad) {
	Tensor output;
	if (last_required_grad) {
		output = Tensor(input_shape);
		assert(delta.shape == output_shape);
		vector<int> new_shape = delta.shape;
		int last_dim = delta.size(-1) * delta.size(-2);
		new_shape.pop_back();
		new_shape.pop_back();
		new_shape.push_back(last_dim);
		delta.reshape(new_shape);
		if (use_bias) {
			gradient_bias = Tensor({ out_channels });
			backward_bias(delta);
		}
		Tensor batch_gradient_weight = delta.matmul(col_input.transpose(1, 2));
		sum_gradient_weight(std::move(batch_gradient_weight));
		gradient_input = weights.transpose(0, 1).matmul(delta);

		int delta_per_img_size = gradient_input.numel() / delta.size(0);
		int input_per_img_size = output.numel() / output.size(0);
		for (int i = 0; i < gradient_input.size(0); i++) {
			float* delta_input = gradient_input.data + i * delta_per_img_size;
			float* delta_output = output.data + i * input_per_img_size;
			IMG2COL::col2img_cpu(delta_output, in_channels, input_shape[2], input_shape[3],
				kernel_size, stride, padding, delta_input);
		}
		gradient_input.clear();
	}
	return std::make_pair(std::move(output), last_required_grad);
}

void ConvLayer::update_paramers(const float lr) {
	if (gradient_weight.numel() == 0)return;
	if (use_bias) {
		for (int i = 0; i < bias_weight.numel(); i++)
			bias_weight.data[i] = bias_weight.data[i] - lr * gradient_bias.data[i];
	}

	for (int i = 0; i < weights.numel(); i++)
		weights.data[i] = weights.data[i] - lr * gradient_weight.data[i];
	gradient_bias.clear();
	gradient_weight.clear();
}