#ifndef ACTIVATE_LAYER_H
#define ACTIVATE_LAYER_H

#include "utils.h"
#include "tensor.h"
#include "model.h"
#include <utility>

class ReLULayer :public Module {
private:
	Tensor input_save;
	void forward_relu(Tensor& input);
	void backward_relu(Tensor& input);
public:
	ReLULayer();
	virtual Tensor forward(Tensor& input);
	virtual pair<Tensor, bool> backward(Tensor& delta, bool last_required_grad);
};

class LeakyReLULayer :public Module {
private:
	Tensor input_save;
	int alpha;
	void forward_leaky_relu(Tensor& input);
	void backward_leaky_relu(Tensor& input);
public:
	LeakyReLULayer(float alpha);
	virtual Tensor forward(Tensor& input);
	virtual pair<Tensor, bool> backward(Tensor& delta, bool last_required_grad);
};

#endif // !ACTIVATE_LAYER_H