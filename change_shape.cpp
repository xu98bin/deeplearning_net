#include "change_shape.h"

Flatten::Flatten(int dim) { this->dim = dim; }
Tensor Flatten::forward(Tensor& input) {
	assert(dim < input.dims());
	input_shape = input.shape;
	int numel = input.numel();
	vector<int> output_shape;
	for (int i = 0; i < dim; i++)
		output_shape.push_back(input.size(i));
	int last_size = input.size(dim);
	for (int i = dim + 1; i < input.dims(); i++)
		last_size *= input.size(i);
	output_shape.push_back(last_size);
	Tensor output = input;
	output.reshape(output_shape);
	return std::move(output);
}

pair<Tensor, bool> Flatten::backward(Tensor& delta, bool last_required_grad) {
	Tensor output = delta;
	output.reshape(input_shape);
	return std::make_pair(std::move(output), last_required_grad);
}

Reshape::Reshape(vector<int> new_shape) {
	this->new_shape = new_shape;
}

Tensor Reshape::forward(Tensor& input) {
	input_shape = input.shape;
	Tensor output = input;
	output.reshape(new_shape);
	return std::move(output);
}

pair<Tensor, bool> Reshape::backward(Tensor& delta, bool last_required_grad) {
	Tensor output = delta;
	output.reshape(input_shape);
	return std::make_pair(std::move(output), last_required_grad);
}

Transpose::Transpose(int dim1, int dim2){
	this->dim1 = dim1;
	this->dim2 = dim2;
}

Tensor Transpose::forward(Tensor& input) {
	Tensor output = input.transpose(dim1, dim2);
	return std::move(output);
}

pair<Tensor, bool> Transpose::backward(Tensor& delta, bool last_required_grad) {
	Tensor output = delta.transpose(dim1, dim2);
	return std::make_pair(std::move(output), last_required_grad);
}