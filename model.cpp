#include "model.h"

void Module::train() {
	training = true;
	for (int i = 0; i < order_dict.size(); i++) {
		order_dict[i].second->train();
	}
}

void Module::eval() {
	training = false;
	for (int i = 0; i < order_dict.size(); i++) {
		order_dict[i].second->eval();
	}
}

void Module::update_paramers(const float lr) {
	return;
}

Tensor Module::forward(Tensor& input) {
	return input;
}

pair<Tensor, bool> Module::backward(Tensor& delta, bool b) {
	return std::make_pair(delta, b);
}