#include "pooling.h"

BasePooling::BasePooling():pooling_size(1), stride(1){}

void BasePooling::set_stride(int stride) { this->stride = stride; }

void BasePooling::set_pooling_size(int pooling_size) { this->pooling_size = pooling_size; }

MaxPooling2D::MaxPooling2D(int pooling_size, int stride) {
	this->stride = stride;
	this->pooling_size = pooling_size;
}

Tensor MaxPooling2D::forward(Tensor& input) {
	assert(input.dims() == 4);
	input_shape = input.shape;
	pooling_idx.clear();
	int out_width = (input.size(3) - pooling_size) / stride + 1;
	int out_height = (input.size(2) - pooling_size) / stride + 1;
	int batch_size = input.numel() / input.size(0);
	int channel_size = batch_size / input.size(1);
	Tensor output({ input.size(0), input.size(1) ,out_height ,out_width });
	int output_batch_size = output.numel() / output.size(0);
	int output_channel_size = out_height * out_width;
	for (int b = 0; b < input.size(0); b++) {
		for (int c = 0; c < input.size(1); c++) {
			for (int h = 0; h < out_height; h++) {
				for (int w = 0; w < out_width; w++) {
					int row = 0, col = 0;
					int start_pos = b * batch_size + c * channel_size + h * stride * input.size(3) + w * stride;
					float max_num = input.data[start_pos];
					for (int i = 0; i < pooling_size; i++) {
						for (int j = 0; j < pooling_size; j++) {
							int tmp_pos = start_pos + i * input.size(3) + j;
							if (input.data[tmp_pos] > max_num) {
								row = i;
								col = j;
								max_num = input.data[tmp_pos];
							}
						}
					}
					int pooling_pos = start_pos + row * input.size(3) + col;
					pooling_idx.push_back(pooling_pos);
					int output_pos = b * output_batch_size + c * output_channel_size + h * out_width + w;
					output.data[output_pos] = max_num;
				}
			}
		}
	}
	return output;
}

pair<Tensor, bool> MaxPooling2D::backward(Tensor& delta, bool last_required_grad) {
	Tensor output(input_shape);
	if (last_required_grad) {
		assert(delta.numel() == pooling_idx.size());
		for (int i = 0; i < delta.numel(); i++) {
			output.data[pooling_idx[i]] += delta.data[i];
		}
	}
	return std::make_pair(output, last_required_grad);
}

AvgPooling2D::AvgPooling2D(int pooling_size, int stride) {
	this->stride = stride;
	this->pooling_size = pooling_size;
}

Tensor AvgPooling2D::forward(Tensor& input) {
	assert(input.dims() == 4);
	input_shape = input.shape;
	int out_width = (input.size(3) - pooling_size) / stride + 1;
	int out_height = (input.size(2) - pooling_size) / stride + 1;
	int batch_size = input.numel() / input.size(0);
	int channel_size = batch_size / input.size(1);
	Tensor output({ input.size(0), input.size(1) ,out_height ,out_width });
	int output_batch_size = output.numel() / output.size(0);
	int output_channel_size = out_height * out_width;
	float sum_pooling = float(pooling_size * pooling_size);
	for (int b = 0; b < input.size(0); b++) {
		for (int c = 0; c < input.size(1); c++) {
			for (int h = 0; h < out_height; h++) {
				for (int w = 0; w < out_width; w++) {
					int row = 0, col = 0;
					int start_pos = b * batch_size + c * channel_size + h * stride * input.size(3) + w * stride;
					float sum_num = 0;
					for (int i = 0; i < pooling_size; i++) {
						for (int j = 0; j < pooling_size; j++) {
							int tmp_pos = start_pos + i * input.size(3) + j;
							sum_num += input.data[tmp_pos];
						}
					}
					int output_pos = b * output_batch_size + c * output_channel_size + h * out_width + w;
					output.data[output_pos] = sum_num / sum_pooling;
				}
			}
		}
	}
	return output;
}

pair<Tensor, bool> AvgPooling2D::backward(Tensor& delta, bool last_required_grad) {
	assert(delta.dims() == 4);
	float sum_pooling = float(pooling_size * pooling_size);
	Tensor output(input_shape);
	output.zerosTensor();
	
	int batch_size = output.numel() / output.size(0);
	int channel_size = batch_size / output.size(1);
	
	int delta_width = delta.size(-1);
	int delta_height = delta.size(-2);
	int delta_batch_size = delta.numel() / delta.size(0);
	int delta_channel_size = delta_height * delta_width;
	if (last_required_grad) {
		for (int b = 0; b < delta.size(0); b++) {
			for (int c = 0; c < delta.size(1); c++) {
				for (int h = 0; h < delta_height; h++) {
					for (int w = 0; w < delta_width; w++) {
						int delta_pos = b * delta_batch_size + c * delta_channel_size + h * delta_width + w;
						int start_pos = b * batch_size + c * channel_size + h * stride * output.size(3) + w * stride;
						for (int i = 0; i < pooling_size; i++) {
							for (int j = 0; j < pooling_size; j++) {
								int output_pos = start_pos + i * output.size(3) + j;;
								output.data[output_pos] += delta.data[delta_pos] / sum_pooling;
							}
						}
					}
				}
			}
		}
	}
	return std::make_pair(output, last_required_grad);
}

AdaptiveAvgPool2D::AdaptiveAvgPool2D(int output_size) { this->output_size = output_size; }

void AdaptiveAvgPool2D::set_stride(int input_size) {
	stride = input_size / output_size;
	if (stride * output_size < input_size)stride++;
}

void AdaptiveAvgPool2D::set_pooling_size(int input_size) {
	pooling_size = input_size - (output_size - 1) * stride;
}

Tensor AdaptiveAvgPool2D::forward(Tensor& input) {
	int input_size = input.size(-1);
	set_stride(input_size);
	set_pooling_size(input_size);
	avgpooling2d.set_pooling_size(pooling_size);
	avgpooling2d.set_stride(stride);
	Tensor output = avgpooling2d.forward(input);
	return std::move(output);
}

pair<Tensor, bool> AdaptiveAvgPool2D::backward(Tensor& delta, bool last_required_grad) {
	return avgpooling2d.backward(delta, last_required_grad);
}

AdaptiveMaxPool2D::AdaptiveMaxPool2D(int output_size) { this->output_size = output_size; }

void AdaptiveMaxPool2D::set_stride(int input_size) {
	stride = input_size / output_size;
	if (stride * output_size < input_size)stride++;
}

void AdaptiveMaxPool2D::set_pooling_size(int input_size) {
	pooling_size = input_size - (output_size - 1) * stride;
}

Tensor AdaptiveMaxPool2D::forward(Tensor& input) {
	int input_size = input.size(-1);
	set_stride(input_size);
	set_pooling_size(input_size);
	maxpooling2d.set_pooling_size(pooling_size);
	maxpooling2d.set_stride(stride);
	Tensor output = maxpooling2d.forward(input);
	return std::move(output);
}

pair<Tensor, bool> AdaptiveMaxPool2D::backward(Tensor& delta, bool last_required_grad) {
	return maxpooling2d.backward(delta, last_required_grad);
}