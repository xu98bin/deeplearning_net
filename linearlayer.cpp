#include "linearlayer.h"

void LinearLayer::mean_gradient_bias(Tensor& delta) {
	if (gradient_bias.numel() == 0)gradient_bias = Tensor({ out_features });
	int last_dim = delta.size(-1);
	int sum_left = delta.numel() / last_dim;
	assert(last_dim == out_features);
	assert(gradient_bias.numel() == out_features);
	for (int i = 0; i < last_dim; i++) {
		for (int j = 0; j < sum_left; j++) {
			gradient_bias.data[i] += delta.data[j * last_dim + i];
		}
	}
	int tmp = sum_left / delta.size(-2);
	for (int i = 0; i < last_dim; i++)
		gradient_bias.data[i] /= float(tmp);
}

void LinearLayer::sum_gradient_bias(Tensor& delta) {
	if (gradient_bias.numel() == 0)gradient_bias = Tensor({ out_features });
	gradient_bias.zerosTensor();
	int last_dim = delta.size(-1);
	int sum_left = delta.numel() / last_dim;
	assert(last_dim == out_features);
	assert(gradient_bias.numel() == out_features);
	for (int i = 0; i < last_dim; i++) {
		for (int j = 0; j < sum_left; j++) {
			gradient_bias.data[i] += delta.data[j * last_dim + i];
		}
	}
}

LinearLayer::LinearLayer(int in_features, int out_features, bool use_bias) {
	this->in_features = in_features;
	this->out_features = out_features;
	weight = Tensor(vector<int>{in_features, out_features});
	weight.normalTensor(0.0f,0.02f);
	if (use_bias)bias = Tensor(vector<int>{out_features});
	this->use_bias = use_bias;
}

Tensor LinearLayer::forward(Tensor& input) {
	input_save = input;
	Tensor output = matmul(input, weight);
	if (use_bias)output.add(bias);
	return std::move(output);
}

void LinearLayer::sum_gradient_weight(Tensor& batch_gradient_weight) {
	if (gradient_weight.numel() == 0)gradient_weight = Tensor(weight.shape);
	int batch_sum = batch_gradient_weight.numel();
	int weight_sum = weight.numel();
	assert(batch_sum % weight_sum == 0);
	int batches = batch_sum / weight_sum;
	for (int b = 0; b < batches; b++) {
		float* tmp = &batch_gradient_weight.data[b * weight_sum];
		for (int i = 0; i < weight_sum; i++)
			gradient_weight.data[i] += tmp[i];
	}
}

void LinearLayer::mean_gradient_weight(Tensor& batch_gradient_weight) {
	if (gradient_weight.numel() == 0)gradient_weight = Tensor(weight.shape);
	int batch_sum = batch_gradient_weight.numel();
	int weight_sum = gradient_weight.numel();
	assert(batch_sum % weight_sum == 0);
	int batches = batch_sum / weight_sum;
	for (int b = 0; b < batches; b++) {
		float* tmp = &batch_gradient_weight.data[b * weight_sum];
		for (int i = 0; i < weight_sum; i++)
			gradient_weight.data[i] += tmp[i];
	}

	for (int i = 0; i < weight_sum; i++)
		gradient_weight.data[i] /= float(batches);
}

pair<Tensor, bool> LinearLayer::backward(Tensor& delta, bool last_required_grad) {
	Tensor gradient_input;
	if (last_required_grad && required_grad) {
		gradient_input = delta.matmul(weight.transpose(0, 1));
		Tensor batch_gradient_weight = input_save.transpose(-1, -2).matmul(delta);
		sum_gradient_weight(batch_gradient_weight);
		if (use_bias)sum_gradient_bias(delta);
	}
	last_required_grad = last_required_grad && required_grad;
	return std::make_pair(std::move(gradient_input), last_required_grad);
}

void LinearLayer::update_paramers(const float lr) {
	if (gradient_weight.numel() == 0)return;
	for (int i = 0; i < bias.numel(); i++)
		bias.data[i] -= lr * gradient_bias.data[i];

	for (int i = 0; i < weight.numel(); i++)
		weight.data[i] -= lr * gradient_weight.data[i];
	gradient_bias.clear();
	gradient_weight.clear();
	input_save.clear();
}