#pragma once
#include "model.h"
#include "tensor.h"
#include <vector>
using std::vector;

class BasePooling :public Module {
public:
	int pooling_size, stride;
public:
	virtual void set_stride(int stride);
	virtual void set_pooling_size(int pooling_size);
	BasePooling();
};

class MaxPooling2D :public BasePooling {
public:
	vector<int> input_shape;
	vector<int> pooling_idx;
public:
	MaxPooling2D() = default;
	MaxPooling2D(int pooling_size, int stride);
	virtual Tensor forward(Tensor& input);
	virtual pair<Tensor, bool> backward(Tensor& delta, bool last_required_grad);
};

class AvgPooling2D :public BasePooling {
public:
	vector<int> input_shape;
public:
	AvgPooling2D()=default;
	AvgPooling2D(int pooling_size, int stride);
	virtual Tensor forward(Tensor& input);
	virtual pair<Tensor, bool> backward(Tensor& delta, bool last_required_grad);
};

class AdaptiveAvgPool2D :public BasePooling {
public:
	AvgPooling2D avgpooling2d;
	int output_size;
	void set_stride(int input_size);
	void set_pooling_size(int pooling_size);
public:
	AdaptiveAvgPool2D(int output_size);
	virtual Tensor forward(Tensor& input);
	virtual pair<Tensor, bool> backward(Tensor& delta, bool last_required_grad);
};

class AdaptiveMaxPool2D :public BasePooling {
public:
	MaxPooling2D maxpooling2d;
	int output_size;
	void set_stride(int input_size);
	void set_pooling_size(int pooling_size);
public:
	AdaptiveMaxPool2D(int output_size);
	virtual Tensor forward(Tensor& input);
	virtual pair<Tensor, bool> backward(Tensor& delta, bool last_required_grad);
};