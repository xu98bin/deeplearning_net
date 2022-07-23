#pragma once
#include <assert.h>
#include <vector>
#include <utility>
#include <string>
#include "tensor.h"
#include "img2col.h"
#include "utils.h"

using std::vector;
using std::pair;
using std::string;


class Module {
public:
	bool training, required_grad;
	vector<pair<string, Module*>> order_dict;

public:
	Module() :training(true), required_grad(true), order_dict() {}
	void train();
	void eval();
	virtual Tensor forward(Tensor& input);
	virtual pair<Tensor,bool> backward(Tensor& delta,bool last_required_grad);
	virtual void update_paramers(const float lr);
};