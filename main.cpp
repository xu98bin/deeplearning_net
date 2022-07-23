//#define _CRTDBG_MAP_ALLOC
#include <crtdbg.h>
#include "model.h"
#include <cstdlib>
#include <vector>
#include "activatelayer.h"
#include "loss.h"
#include "linearlayer.h"
#include "pooling.h"
#include "convlayer.h"
#include "LeNet.h"
#include <opencv2/opencv.hpp>
using namespace std;
#define BYTE unsigned char
#define tuple4 std::tuple<vector<vector<BYTE>>, vector<BYTE>, vector<vector<BYTE>>, vector<BYTE>>

void test_cross_entroy_loss() {
	Tensor input({ 32,10 });
	input.normalTensor(0.0f, 1.0f);
	cout << "input:" << input << endl;

	Tensor labels({ 32 });
	for (int i = 0; i < 32; i++) {
		labels.data[i] = float(rand() % 10);
	}
	cout << "labels:" << labels << endl;

	CrossEntropyLoss loss;
	loss.set_labels(labels);
	Tensor output_loss = loss.forward(input);
	cout << "output_loss" << output_loss << endl;
	Tensor delta({ 1 });
	delta.onesTensor();
	pair<Tensor, bool> ans = loss.backward(delta, true);
	cout << "gradient" << ans.first << endl;
}

void test_linear() {
	Tensor input({ 4,16,64 });
	input.normalTensor(0.0f, 2.0f);
	cout << "input:"<<input << endl;
	LinearLayer linear1(64, 32, true);
	linear1.weight.normalTensor(0.0f, 0.02f);
	cout << "weight" << linear1.weight << endl;
	linear1.bias.normalTensor(0.0f, 0.02f);
	cout << "bias" << linear1.bias << endl;
	Tensor output=linear1.forward(input);
	cout << "output:" << output << endl;
	Tensor delta(output.shape);
	delta.onesTensor();
	pair<Tensor, bool> ans = linear1.backward(delta, true);
	cout << "weight_gradient:" << linear1.gradient_weight << endl;
	cout << "gradient_bias:" << linear1.gradient_bias << endl;
}

void test_maxpooling() {
	Tensor input({4,4,16,16 });
	input.normalTensor(0.0f, 1.0f);
	cout << "input:" << input << endl;
	MaxPooling2D maxpool(3, 2);
	Tensor output = maxpool.forward(input);
	cout << "output:" << output << endl;
	Tensor delta({ 4,4,7,7 });
	delta.uniformTensor(0.0f, 1.0f);
	cout << "delta:" << delta << endl;
	pair<Tensor, bool> ans = maxpool.backward(delta, true);
	cout << "gradient:" << ans.first << endl;
}

void test_avgpooling() {
	Tensor input({ 4,4,16,16 });
	input.normalTensor(0.0f, 1.0f);
	cout << "input:" << input << endl;
	AvgPooling2D avgpool(3, 2);
	Tensor output = avgpool.forward(input);
	cout << "output:" << output << endl;
	Tensor delta({ 4,4,7,7 });
	delta.uniformTensor(0.0f, 1.0f);
	cout << "delta:" << delta << endl;
	pair<Tensor, bool> ans = avgpool.backward(delta, true);
	cout << "gradient:" << ans.first << endl;
}

void test_adaptiveMaxpooling() {
	Tensor input({ 4,4,16,16 });
	input.normalTensor(0.0f, 1.0f);
	cout << "input:" << input << endl;
	AdaptiveMaxPool2D avgpool(2);
	Tensor output = avgpool.forward(input);
	cout << "output:" << output << endl;
	Tensor delta({ 4,4,2,2 });
	delta.uniformTensor(0.0f, 1.0f);
	cout << "delta:" << delta << endl;
	pair<Tensor, bool> ans = avgpool.backward(delta, true);
	cout << "gradient:" << ans.first << endl;
}

void test_adaptiveAvgpooling() {
	Tensor input({ 4,4,16,16 });
	input.normalTensor(0.0f, 1.0f);
	cout << "input:" << input << endl;
	AdaptiveAvgPool2D avgpool(1);
	Tensor output = avgpool.forward(input);
	cout << "output:" << output << endl;
	Tensor delta({ 4,4,1,1 });
	delta.uniformTensor(0.0f, 1.0f);
	cout << "delta:" << delta << endl;
	pair<Tensor, bool> ans = avgpool.backward(delta, true);
	cout << "gradient:" << ans.first << endl;
}

void test_conv() {
	Tensor input({ 4,3,16,16 });
	input.normalTensor(0.0f, 1.0f);
	cout << "input:" << input << endl;
	ConvLayer conv(3, 2, 1, 3, 12, true);
	conv.weights.normalTensor(0.0f, 0.02f);
	cout << "weight:" << conv.weights << endl;
	conv.bias_weight.normalTensor(0.0f, 1.0f);
	cout << "bias_weight:" << conv.bias_weight << endl;
	Tensor output = conv.forward(input);
	cout << "output:" << output << endl;
	Tensor delta({ 4,12,8,8 });
	delta.normalTensor(0.0f, 0.2f);
	cout << "delta:" << delta << endl;
	pair<Tensor, bool> ans = conv.backward(delta, true);
	//cout << "gradient:" << ans.first << endl;
	cout << "weights_gradient:" << conv.gradient_weight << endl;
	cout << "bias_gradient:" << conv.gradient_bias<< endl;
}

void test_re() {
	Tensor tensor1({ 1, 1, 28, 28 });
	tensor1.uniformTensor(-1.0f, 1.0f);
	std::cout << "tensor1" << tensor1 << std::endl;
	ReLULayer re;
	Tensor output = re.forward(tensor1);
	cout << "re_output" << output << endl;
	Tensor delta({ 1, 1, 28, 28 });
	delta.normalTensor(0.0f, 0.2f);
	cout << "delta:" << delta << endl;
	pair<Tensor, bool> ans = re.backward(delta, true);
	cout << "input_grad:" << ans.first << endl;
}

void test_mnist() {
	LeNet lenet;
	lenet.start_train("C:\\Users\\23799\\Desktop\\mnist");
}

void test_matmul() {
	Tensor tensor1({ 4,4,6,6 });
	tensor1.normalTensor(0.0f, 0.2f);
	std::cout << "tensor1" << tensor1 << std::endl;
	Tensor tensor2({ 4,4,6,6 });
	tensor2.normalTensor(0.0f, 0.2f);
	std::cout << "tensor2" << tensor2 << std::endl;
	tensor1.matmul_(tensor2);
	std::cout << "tensor1" << tensor1 << std::endl;
}

void test_conv_act_linear() {
	Tensor tensor1({ 1, 1, 28, 28 });
	tensor1.uniformTensor(0.0f, 1.0f);
	std::cout << "tensor1" << tensor1 << std::endl;
	ConvLayer conv(3, 2, 1, 1, 1, true);
	conv.weights.normalTensor(0.0f, 0.02f);
	//cout << "weight:" << conv.weights << endl;
	conv.bias_weight.normalTensor(0.0f, 1.0f);
	//cout << "bias_weight:" << conv.bias_weight << endl;
	Tensor output = conv.forward(tensor1);
	//cout << "conv_output:" << output << endl;
	ReLULayer re;
	output = re.forward(output);
	//cout << "re_output" << output << endl;
	Reshape reshape(vector<int>{1, 1, 14 * 14});
	output = reshape.forward(output);
	//cout << "reshape_output" << output << endl;
	LinearLayer linear(14 * 14, 10, true);
	linear.weight.normalTensor(0.0f, 0.2f);
	//cout << "linear_weight" << linear.weight << endl;
	linear.bias.zerosTensor();
	//cout << "linear_bias" << linear.bias<< endl;
	output = linear.forward(output);
	//cout << "linear_output" << output << endl;
	Tensor delta({ 1,1,10 });
	delta.normalTensor(0.0f, 0.2f);
	cout << "delta:" << delta << endl;
	pair<Tensor, bool> ans = linear.backward(delta, true);
	cout << "linear_weight_grad:" << linear.gradient_weight << endl;
	cout << "linear_bias_grad:" << linear.gradient_bias << endl;
	ans = reshape.backward(ans.first, ans.second);
	ans = re.backward(ans.first, ans.second);
	ans = conv.backward(ans.first, ans.second);
	cout << "conv_weight_grad:" << conv.gradient_weight << endl;
	cout << "conv_bias_grad:" << conv.gradient_bias << endl;
	cout << "input_grad:" << ans.first << endl;
}

void test_lenet() {
	Tensor tensor1({ 1, 1, 28, 28 });
	tensor1.uniformTensor(0.0f, 1.0f);
	std::cout << "tensor1" << tensor1 << std::endl;
	ConvLayer conv1 = ConvLayer(5, 1, 2, 1, 5, true);
	ConvLayer conv2 = ConvLayer(5, 2, 2, 5, 10, true);
	MaxPooling2D max_pool1 = MaxPooling2D(2, 2);
	MaxPooling2D max_pool2 = MaxPooling2D(2, 2);

}

void test_image() {
	MNIST dataset("C:\\Users\\23799\\Desktop\\mnist");
	tuple4 train_and_test = dataset.image2vector();
	vector<vector<BYTE>> train_imgs = std::get<0>(train_and_test);
	vector<BYTE> train_labels = std::get<1>(train_and_test);
	vector<vector<BYTE>> test_imgs = std::get<2>(train_and_test);
	vector<BYTE> test_labels = std::get<3>(train_and_test);
	for (int i = 0; i < train_imgs.size() / (32*32) * (32*32); i += (32 * 32)) {
		cv::Mat image = cv::Mat::zeros(32 * 28, 32 * 28, CV_8UC1);
		std::vector<BYTE> label(train_labels.begin() + i, train_labels.begin() + i + 32 * 32);
		for (int j = 0; j < 32; j++) {
			for (int k = 0; k < 32; k++) {
				for (int m = 0; m < 28; m++) {
					for (int n = 0; n < 28; n++) {
						int row = j * 28 + m;
						int col = k * 28 + n;
						int or_pos = i + j * 32 + k;
						image.at<unsigned char>(row, col) = train_imgs[or_pos][m * 28 + n];
					}
				}
			}
		}
		for (int i = 0; i < 32; i++)
			for (int j = 0; j < 32; j++) {
				std::cout << int(label[i * 32 + j]);
				if (j == 31)std::cout << std::endl;
			}
		cv::imshow("image", image);
		int key = cv::waitKey();
		if (key == int('q')) {
			exit(-1);
		}
	}
}

int main(int argc, char** argv) {
	test_mnist();
	//_CrtDumpMemoryLeaks();
	//test_conv_act_linear();
	//test_image();
}