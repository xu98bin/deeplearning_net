#pragma once
#include "model.h"
#include "tensor.h"

class Flatten :public Module {
public:
	vector<int> input_shape;
	int dim;
public:
	Flatten(int dim);
	virtual Tensor forward(Tensor& input);
	virtual pair<Tensor, bool> backward(Tensor& delta, bool last_required_grad);
};

class Reshape :public Module {
public:
	vector<int> input_shape,new_shape;
public:
	Reshape(vector<int> new_shape);
	virtual Tensor forward(Tensor& input);
	virtual pair<Tensor, bool> backward(Tensor& delta, bool last_required_grad);
};

class Transpose :public Module {
public:
	int dim1, dim2;
public:
	Transpose(int dim1, int dim2);
	virtual Tensor forward(Tensor& input);
	virtual pair<Tensor, bool> backward(Tensor& delta, bool last_required_grad);
};