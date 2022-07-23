#pragma once
#include "tensor.h"
#include "model.h"

class LinearLayer :public Module {
private:
	void sum_gradient_bias(Tensor& delta);
	void mean_gradient_bias(Tensor& delta);
	void sum_gradient_weight(Tensor& batch_gradient_weight);
	void mean_gradient_weight(Tensor& batch_gradient_weight);
public:
	int in_features, out_features;
	bool use_bias;
	Tensor weight, bias, gradient_weight, gradient_bias, input_save;
	LinearLayer(int in_features, int out_features, bool use_bias);
	virtual Tensor forward(Tensor& input);
	virtual pair<Tensor, bool> backward(Tensor& delta, bool last_required_grad);
	virtual void update_paramers(const float lr);
};