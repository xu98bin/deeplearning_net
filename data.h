#pragma once
#include "tensor.h"
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include "tensor.h"
#include <deque>
#include <atomic>


class BaseDataset {
public:
	BaseDataset(){}
	virtual size_t len() = 0;
	virtual std::vector<Tensor> getitem(int index) = 0;
	virtual ~BaseDataset() {};
};

typedef std::vector<Tensor>(*fun)(vector<int>, BaseDataset*);

class BaseSampler {
protected:
	std::vector<int> indexs;
	std::mutex _mutex;
public:
	~BaseSampler(){}
	virtual void init(int length) = 0;

	bool empty() {
		std::lock_guard<std::mutex> lock(_mutex);
		return indexs.empty();
	}

	int enumrate() {
		std::lock_guard<std::mutex> lock(_mutex);
		if (indexs.size() == 0)return -1;
		int ans = indexs[0];
		indexs.erase(indexs.begin());
		return ans;
	}

	std::vector<int> enumrate(int batch_size) {
		std::lock_guard<std::mutex> lock(_mutex);
		std::vector<int> ans;
		for (int i = 0; i < batch_size; i++) {
			if (indexs.size() == 0)ans.push_back(-1);
			else {
				ans.push_back(indexs[0]);
				indexs.erase(indexs.begin());
			}
		}
		return ans;
	}
};

class ShuffleSampler :public BaseSampler {
public:
	void init(int length) {
		std::vector<int> tmp;
		for (int i = 0; i < length; i++)
			tmp.push_back(i);
		for (int i = 0; i < length; i++) {
			int pos = rand() % (length - i);
			indexs.push_back(tmp[pos]);
			tmp.erase(tmp.begin() + pos);
		}
	}
};

class SortedSampler :public BaseSampler {
public:
	void init(int length) {
		std::vector<int> tmp;
		for (int i = 0; i < length; i++)
			tmp.push_back(i);
	}
};

std::vector<Tensor> collate_fn(vector<int> indexs, BaseDataset* dataset) {
	for (int i = indexs.size() - 1; i >= 0; i--)
		if (indexs[i] == -1)indexs.pop_back();
	int c, h, w;
	int batch_size = indexs.size();
	Tensor output_image_tensor, output_labels_tensor;
	for (int i = 0; i < indexs.size(); i++) {
		vector<Tensor> img_label = dataset->getitem(indexs[i]);
		if (i == 0) {
			h = img_label[0].size(-2);
			w = img_label[0].size(-1);
			c = img_label[0].size(0);
			output_image_tensor = Tensor({ batch_size, c, h, w});
			output_labels_tensor = Tensor({ batch_size });
		}
		else {
			if (h != img_label[0].size(-2)) {
				printf_s("input image's h is not equal!\n");
				exit(-1);
			}
			if (w == img_label[0].size(-1)) {
				printf_s("input image's w is not equal!\n");
				exit(-1);
			}
			if (c == img_label[0].size(0)) {
				printf_s("input image's channel is not equal!\n");
				exit(-1);
			}
		}
		memcpy(&output_image_tensor.data[i * c * h * w], img_label[0].data, c * h * w);
		output_labels_tensor.data[i] = img_label[1].data[0];
	}
	return std::vector<Tensor>{std::move(output_image_tensor), std::move(output_labels_tensor)};
}

class TensorDeque {
private:
	std::deque<vector<Tensor>> _queue;
	std::mutex _mutex;
public:
	void push_back(vector<Tensor>&& tensor) {
		std::lock_guard<std::mutex> lock(_mutex);
		_queue.push_back(tensor);
	}

	vector<Tensor> pop_front() {
		std::lock_guard<std::mutex> lock(_mutex);
		vector<Tensor> output;
		if (!_queue.empty()) {
			output = _queue[0];
			_queue.pop_front();
		}
		if (output.size())
			return std::move(output);
		else return output;
	}

	int size() {
		std::lock_guard<std::mutex> lock(_mutex);
		return _queue.size();
	}

	bool empty() {
		std::lock_guard<std::mutex> lock(_mutex);
		return _queue.empty();
	}
};

class BaseDataloader {
private:
	BaseDataset* this_dataset;
	TensorDeque task_queue;
	std::vector<std::thread> worker_threads;
	int task_size, num_workers, batch_size;
	bool drop_last, shuffle, finished;
	std::atomic_bool finished;
	size_t loader_size;
	int index;
	BaseSampler* sampler;
	std::condition_variable cond1, cond2;
	std::mutex _mutex1, _mutex2;
private:
	void read_data() {
		while (finished == false) {
			std::unique_lock<std::mutex> lock(_mutex1);
			vector<int> indexs = sampler->enumrate(batch_size);
			if (indexs[indexs.size() - 1] == -1)finished = true;
			if (indexs[0] == -1)break;
			std::vector<Tensor> ans = std::move(collate_fn(indexs, this_dataset));
			if (task_queue.size() >= task_size) {
				cond1.wait(lock);
			}
			if (task_queue.empty()) {
				task_queue.push_back(std::move(ans));
				cond2.notify_one();
			}
			else {
				task_queue.push_back(std::move(ans));
			}
		}
	}

public:
	BaseDataloader(BaseDataset* dataset, BaseSampler* sampler, int batch_size, int num_workers, bool drop_last, bool shuffle):
		this_dataset(dataset),sampler(sampler){
		this->drop_last = drop_last;
		this->shuffle = shuffle;
		this->num_workers = max(num_workers, std::thread::hardware_concurrency());
		this->task_size = this->num_workers;
		this->batch_size = batch_size;
		loader_size = size();
		finished = false;
	}

	void prepare() {
		sampler->init(this_dataset->len());
		finished = false;
		for (int i = 0; i < num_workers; i++)
			worker_threads.push_back(std::thread(&BaseDataloader::read_data, this));
	}

	std::vector<Tensor> enumrate() {
		std::unique_lock<std::mutex> lock(_mutex2);
		if (task_queue.empty()&&finished==false)cond2.wait(lock);
		std::vector<Tensor> ans;
		if (task_queue.size() >= task_size) {
			ans = task_queue.pop_front();
			cond1.notify_one();
		}
		else {
			ans = task_queue.pop_front();
		}
		if (finished == true) {
			for (int i = 0; i < num_workers; i++)
				if (worker_threads[i].joinable())
					worker_threads[i].join();
		}
		return std::move(ans);
	}

	size_t size() {
		int length = this_dataset->len();
		if (length / batch_size * batch_size < length)return length / batch_size + 1;
		else return length / batch_size;
	}
};