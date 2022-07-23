#include "activatelayer.h"
#include "utils.h"

ReLULayer::ReLULayer() {}

void ReLULayer::forward_relu(Tensor& input) {
	int sum = input.numel();
	for (int i = 0; i < sum; i++) {
		input.data[i] = input.data[i] > 0.0f ? input.data[i] : 0.0f;
	}
}

void ReLULayer::backward_relu(Tensor& input) {
	int sum = input.numel();
	for (int i = 0; i < sum; i++) {
		input.data[i] = input.data[i] > 0.0f ? 1.0f : 0.0f;
	}
}

Tensor ReLULayer::forward(Tensor& input) {
	input_save = input;
	Tensor output = input;
	forward_relu(output);
	return std::move(output);
}

pair<Tensor, bool> ReLULayer::backward(Tensor& delta, bool last_required_grad) {
	Tensor output = input_save;
	if (last_required_grad) {
		backward_relu(output);
	}
	output.dot_(delta);
	return std::make_pair(std::move(output), last_required_grad);
}


LeakyReLULayer::LeakyReLULayer(float alpha) { this->alpha = alpha; }

void LeakyReLULayer::forward_leaky_relu(Tensor& input) {
	int sum = input.numel();
	for (int i = 0; i < sum; i++) {
		input.data[i] = input.data[i] > 0.0f ? input.data[i] : input.data[i] * alpha;
	}
}

void LeakyReLULayer::backward_leaky_relu(Tensor& input) {
	int sum = input.numel();
	for (int i = 0; i < sum; i++) {
		input.data[i] = input.data[i] > 0.0f ? 1.0f : alpha;
	}
}

Tensor LeakyReLULayer::forward(Tensor& input) {
	input_save = input;
	Tensor output = input;
	forward_leaky_relu(output);
	return std::move(output);
}

pair<Tensor, bool> LeakyReLULayer::backward(Tensor& delta, bool last_required_grad) {
	Tensor output = input_save;
	if (last_required_grad) {
		backward_leaky_relu(output);
	}
	output.dot_(delta);
	return std::make_pair(std::move(output), last_required_grad);
}