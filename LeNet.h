#pragma once
#include "model.h"
#include "linearlayer.h"
#include "pooling.h"
#include "convlayer.h"
#include "change_shape.h"
#include <vector>
#include <utility>
#include <thread>
#include <mutex>
#include "data_ops.h"
#include <tuple>
#include "loss.h"
#include "mycalloc.h"
#include <time.h>
using std::vector;
using std::shared_ptr;
#define BYTE unsigned char
#define tuple4 std::tuple<vector<vector<BYTE>>, vector<BYTE>, vector<vector<BYTE>>, vector<BYTE>>

class LeNet:public Module {
public:
	vector<shared_ptr<Module>> model_ft;
	
	LeNet() {
		shared_ptr<Module> model1(new ConvLayer(5, 1, 2, 1, 8, true));
		shared_ptr<Module> model2(new MaxPooling2D(2, 2));
		shared_ptr<Module> model3(new ReLULayer());
		shared_ptr<Module> model4(new ConvLayer(5, 1, 2, 8, 32, true));
		shared_ptr<Module> model5(new MaxPooling2D(2, 2));
		shared_ptr<Module> model6(new ReLULayer());
		shared_ptr<Module> model7(new Flatten(1));
		shared_ptr<Module> model8(new LinearLayer(1568, 128, true));
		shared_ptr<Module> model9(new ReLULayer());
		shared_ptr<Module> model10(new LinearLayer(128, 10, true));
		model_ft.push_back(model1);
		model_ft.push_back(model2);
		model_ft.push_back(model3);
		model_ft.push_back(model4);
		model_ft.push_back(model5);
		model_ft.push_back(model6);
		model_ft.push_back(model7);
		model_ft.push_back(model8);
		model_ft.push_back(model9);
		model_ft.push_back(model10);
	}

	Tensor forward(Tensor& input) {
		Tensor tmp = input;
		for (int i = 0; i < model_ft.size(); i++)
			tmp = model_ft[i]->forward(tmp);
		return std::move(tmp);
	}

	pair<Tensor, bool> backward(Tensor& delta, bool last_required_grad) {
		pair<Tensor, bool> ans(delta, last_required_grad);
		for (int i = model_ft.size() - 1; i >= 0; i--)
			ans = model_ft[i]->backward(ans.first, ans.second);
		return ans;
	}

	void update_paramers(float lr) {
		for (int i = 0; i < model_ft.size(); i++)
			model_ft[i]->update_paramers(lr);
	}

	void load_tensor(Tensor& tensor_imgs, vector<vector<BYTE>>& imgs, Tensor& tensor_labels, vector<BYTE>& labels,int start_pos) {
		for (int i = 0; i < 128; i++) {
			float* tensor_img = tensor_imgs.data + i * 784;
			tensor_labels.data[i] = float(labels[i + start_pos]);
			vector<BYTE> tmp = imgs[i + start_pos];
			for (int j = 0; j < 784; j++) {
				tensor_img[j] = float(tmp[j]) / 255.0f;
			}
		}
	}

	vector<BYTE> get_max(Tensor& output_labels) {
		vector<BYTE> ans;
		for (int i = 0; i < output_labels.size(0); i++) {
			float* labels_start = output_labels.data + i * 10;
			BYTE pos = 0;
			float num_max = labels_start[0];
			for (BYTE j = 1; j < 10; j++) {
				if (labels_start[int(j)] > num_max) {
					num_max = labels_start[int(j)];
					pos = j;
				}
			}
			ans.push_back(pos);
		}
		return std::move(ans);
	}

	float precision(vector<BYTE>& pred, vector<BYTE>& real) {
		float ans = 0.0f;
		assert(pred.size() <= real.size());
		for (int i = 0; i < pred.size(); i++) {
			if (pred[i] == real[i])ans++;
		}
		float length = float(pred.size());
		return ans / length;
	}

	void start_train(std::string MINST_PATH, int epoches = 10) {
		MNIST dataset(MINST_PATH);
		tuple4 train_and_test = dataset.image2vector();
		vector<vector<BYTE>> train_imgs = std::get<0>(train_and_test);
		vector<BYTE> train_labels = std::get<1>(train_and_test);
		vector<vector<BYTE>> test_imgs = std::get<2>(train_and_test);
		vector<BYTE> test_labels = std::get<3>(train_and_test);
		Tensor input_imgs({ 128,1, 28,28});
		Tensor input_labels({ 128 });
		Tensor output;
		CrossEntropyLoss loss_fnc;
		Tensor loss({ 1 });
		Tensor delta({ 1 });
		delta.data[0] = 1.0f;
		for (int i = 0; i < epoches; i++) {
			float train_sum_loss = 0.0f;
			float prec = 0.0f;
			int hhh = 0;
			vector<BYTE> labels, pred_labels;
			clock_t begin = clock();
			for (int j = 0; j < train_labels.size() / 128; j++) {
				//input_imgs.reshape({ 128, 28, 28, 1 });
				int start_pos = j * 128;
				load_tensor(input_imgs, train_imgs, input_labels, train_labels, start_pos);
				//input_imgs.transpose_(1, 3);
				//input_imgs.transpose_(2, 3);
				//std::cout << input_imgs << std::endl;
				output = forward(input_imgs);
				vector<BYTE> ans = get_max(output);
				for (const BYTE& b : ans)
					pred_labels.push_back(b);
				vector<BYTE> tmp_labels(train_labels.begin() + start_pos, train_labels.begin() + start_pos + 128);
				for (const BYTE& b : tmp_labels)
					labels.push_back(b);
				loss_fnc.set_labels(input_labels);
				loss = loss_fnc.forward(output);
				train_sum_loss += loss.data[0];
				pair<Tensor,bool> tmp_pair=loss_fnc.backward(delta, true);
				backward(tmp_pair.first, true);
				if(i<5)
					update_paramers(0.001);
				else update_paramers(0.0001);
			}
			float train_mean_loss = train_sum_loss / float(labels.size());
			prec = precision(labels, pred_labels);
			clock_t end = clock();
			std::cout << "train  |  epoch=" << i << ";imgs=" << labels.size() << ";precision=" << prec << ";mean_loss=" << train_mean_loss << ";use_time=" << end - begin << std::endl;

			vector<BYTE> test_pred;
			float test_sum_loss = 0.0f;
			for (int j = 0; j < test_imgs.size() / 128; j++) {
				int start_pos = j * 128;
				load_tensor(input_imgs, test_imgs, input_labels, test_labels, start_pos);
				output = forward(input_imgs);
				vector<BYTE> ans = get_max(output);
				for (BYTE& l : ans)test_pred.push_back(l);
				loss_fnc.set_labels(input_labels);
				loss = loss_fnc.forward(output);
				test_sum_loss += loss.data[0];
			}
			prec = precision(test_pred, test_labels);
			std::cout << "test  |  precision=" << prec  <<"  |  num_images="<< test_pred.size() << std::endl;
		}
	}
};