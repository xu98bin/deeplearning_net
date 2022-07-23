#ifndef LOSS_H
#define LOSS_H

#include "model.h"
#include "tensor.h"

class SoftMax :public Module {
public:
	Tensor softmax_save;
public:
	Tensor forward(Tensor& input);
	pair<Tensor, bool> backward(Tensor& delta, bool last_required_grad);
	void update_paramers(float lr);
};

class CrossEntropyLoss :public Module {
private:
	Tensor labels;
	SoftMax softmax;
	Tensor input_save;
public:
	void set_labels(Tensor& other_labels);
	Tensor forward(Tensor& input);
	pair<Tensor, bool> backward(Tensor& delta, bool last_required_grad);
	void update_paramers(float lr);
};

#endif // !LOSS_H