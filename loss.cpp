#include "loss.h"
#include <math.h>
#include <iostream>
using std::cout;
using std::endl;

Tensor SoftMax::forward(Tensor& input) {
	softmax_save = input;
	int batches = input.numel() / input.size(-1);
	Tensor keep_sum_exp = Tensor({ batches });
	keep_sum_exp.zerosTensor();
	Tensor output(input.shape);
	int batch_size = input.size(-1);
	for (int b = 0; b < batches; b++) {
		for (int i = 0; i < batch_size; i++) {
			int pos = b * batch_size + i;
			softmax_save.data[pos] = ::exp(input.data[pos]);
			keep_sum_exp.data[b] += softmax_save.data[pos];
		}
		for (int i = 0; i < batch_size; i++) {
			int pos = b * batch_size + i;
			output.data[pos] = softmax_save.data[pos] / keep_sum_exp.data[b];
		}
	}
	softmax_save = output;
	return std::move(output);
}

pair<Tensor, bool> SoftMax::backward(Tensor& delta, bool last_required_grad) {
	vector<int> out_shape = softmax_save.shape;
	out_shape.push_back(softmax_save.size(-1));
	Tensor output(out_shape);

	int batches = softmax_save.numel() / softmax_save.size(-1);
	int batch_size = softmax_save.size(-1) * softmax_save.size(-1);
	int sqrt_batch_size = softmax_save.size(-1);

	for (int b = 0; b < batches; b++) {
		for (int i = 0; i < sqrt_batch_size; i++) {
			for (int j = 0; j < sqrt_batch_size; j++) {
				int pos = b * batch_size + i * sqrt_batch_size + j;
				int pos_i = b * sqrt_batch_size + i;
				int pos_j = b * sqrt_batch_size + j;
				if (i == j)output.data[pos] = softmax_save.data[pos_i]*(1.0f - softmax_save.data[pos_i]);
				else output.data[pos] = -softmax_save.data[pos_i]* softmax_save.data[pos_j];
			}
		}
	}
	output = delta.matmul(output);
	assert(output.numel() == softmax_save.numel());
	output.shape.pop_back();
	output.shape.pop_back();
	output.shape.push_back(softmax_save.size(-1));
	return std::make_pair(std::move(output), last_required_grad);
}

void SoftMax::update_paramers(float lr) {
	return;
}

void CrossEntropyLoss::set_labels(Tensor& other_labels) {
	assert(other_labels.dims() == 1);
	labels = other_labels;
}

Tensor CrossEntropyLoss::forward(Tensor& input) {
	input_save = softmax.forward(input);
	assert(input.dims() == 2);
	assert(input.size(0) == labels.size(0));
	int tmp = input.size(1);
	Tensor output({ 1 });
	output.zerosTensor();
	for (int b = 0; b < input.size(0); b++) {
		assert(int(labels.data[b]) < tmp);
		output.data[0] -= ::log(input_save.data[b * tmp + int(labels.data[b])]);
	}

	output.data[0] /= float(labels.size(0));
	return std::move(output);
}

pair<Tensor, bool> CrossEntropyLoss::backward(Tensor& delta, bool last_required_grad) {
	assert(delta.numel() == 1);
	float init_gradient = delta.data[0];
	Tensor output = input_save;
	int shape_last = output.size(-1);
	for (int i = 0; i < labels.numel(); i++) {
		int pos = i * shape_last + int(labels.data[i]);
		output.data[pos] -= 1;
	}
	input_save.clear();
	return std::make_pair(std::move(output), true);
}

void CrossEntropyLoss::update_paramers(float lr) {
	return;
}