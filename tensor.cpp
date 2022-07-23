#include "tensor.h"
#include <iomanip>
#include <string.h>
#include <sstream>
#include "cpuinfo.h"
#include "mycalloc.h"
int min(int a, int b) {
    return a > b ? b : a;
}

int max(int a, int b) {
    return a > b ? a : b;
}

int Tensor::dims() const {
    return this->shape.size();
}

int Tensor::size(int dim) const {
    int length = this->dims();
    int pos = 0;
    if (length == 0)return 0;
    if (dim < 0)pos = length + dim;
    else pos = dim;
    return this->shape[pos];
}

int Tensor::numel() const {
    if (dims() == 0)return 0;
    int ans = 1;
    for (const int& i : this->shape)
        ans *= i;
    return ans;
}

void Tensor::reshape(vector<int>& new_shape) {
    int sum = numel();
    int new_sum = new_shape.empty() ? 0 : 1;
    for (int& i : new_shape)
        new_sum *= i;
    assert(new_sum == sum);
    this->shape = new_shape;
}

void Tensor::reshape(vector<int>&& new_shape) {
    reshape(new_shape);
}

Tensor::~Tensor() {
    this->shape.clear();
    if (this->data != nullptr)XCalloc::xfree(this->data);
    this->data = nullptr;
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this == &other)return *this;
    int sum = other.numel();
    if (numel() == sum) {
        this->shape = other.shape;
        this->data = (float*)memcpy(this->data, other.data, sum * sizeof(float));
        return *this;
    }
    float* new_data = (float*)XCalloc::xcalloc(sum, sizeof(float));
    this->clear();
    this->shape = other.shape;
    this->data = (float*)memcpy(new_data, other.data, sum * sizeof(float));
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other){
    assert(this != &other);
    this->clear();
    this->shape = other.shape;
    this->data = other.data;
    other.shape.clear();
    other.data = nullptr;
    return *this;
}

Tensor::Tensor() :data(nullptr) {}

Tensor::Tensor(const Tensor& other) {
    if (this == &other)return;
    int sum = other.numel();
    float* new_data = (float*)XCalloc::xcalloc(sum, sizeof(float));
    assert(new_data != nullptr);
    this->shape = other.shape;
    data = (float*)memcpy(new_data, other.data, sum * sizeof(float));
}

Tensor::Tensor(Tensor&& other) {
    assert(this != &other);//
    this->shape = other.shape;
    this->data = other.data;
    other.data = nullptr;
}

Tensor::Tensor(vector<int>& shape) {
    this->shape = shape;
    int sum = numel();
    data = (float*)XCalloc::xcalloc(sum, sizeof(float));
}

Tensor::Tensor(vector<int>&& shape) :Tensor(shape) {}

Tensor::Tensor(vector<int>& shape, vector<float>& datas) {
    this->shape = shape;
    int sum = numel();
    int length = min(sum, datas.size());
    this->data = (float*)XCalloc::xcalloc(sum, sizeof(float));
    assert(this->data != nullptr);
    for (int i = 0; i < length; i++)
        this->data[i] = datas[i];
    for (int i = length; i < sum; i++)
        this->data[i] = 0.0f;
}

Tensor::Tensor(vector<int>& shape, vector<float>&& datas) :Tensor(shape, datas) {}

Tensor::Tensor(vector<int>& shape, std::initializer_list<float>& datas) {
    this->shape = shape;
    int sum = numel();
    int length = min(sum, datas.size());
    auto begin = datas.begin();
    this->data = (float*)XCalloc::xcalloc(sum, sizeof(float));
    assert(this->data != nullptr);
    for (int i = 0; i < length; i++)
        this->data[i] = *(begin + i);
    for (int i = length; i < sum; i++)
        this->data[i] = 0.0f;
}

Tensor::Tensor(vector<int>& shape, std::initializer_list<float>&& datas) :Tensor(shape, datas) {}

void Tensor::uniformTensor(float min, float max) {
    std::uniform_real_distribution<float> u(min, max);
    std::default_random_engine e;
    int nums = numel();
    for (int i = 0; i < nums; i++)
        data[i] = u(e);
}

void Tensor::uniformTensor(float min, float max, vector<int>& shape) {
    std::uniform_real_distribution<float> u(min, max);
    std::default_random_engine e;
    int nums = 1;
    for (int& i : shape)
        nums *= i;
    if (shape.size() == 0)nums = 0;
    if (nums == numel())return uniformTensor(min, max);
    float* data = (float*)XCalloc::xcalloc(nums, sizeof(float));
    assert(data != nullptr);
    for (int i = 0; i < nums; i++)
        data[i] = u(e);
    if (this->data != nullptr) {
        XCalloc::xfree(this->data);
        this->data = nullptr;
    }
    this->shape = shape;
    this->data = data;
}

void Tensor::uniformTensor(float min, float max, vector<int>&& shape) {
    uniformTensor(min, max, shape);
}

void Tensor::normalTensor(float mean, float sigma) {
    std::normal_distribution<float> u(mean, sigma);
    std::default_random_engine e;
    int sum = numel();
    for (int i = 0; i < sum; i++)
        data[i] = u(e);
}

void Tensor::normalTensor(float mean, float sigma, vector<int>& shape) {
    std::normal_distribution<float> u(mean, sigma);
    std::default_random_engine e;
    int nums = 1;
    for (int& i : shape)
        nums *= i;
    if (shape.size() == 0)nums = 0;
    if (nums == numel())return normalTensor(mean, sigma);
    float* data = (float*)XCalloc::xcalloc(nums, sizeof(float));
    assert(data != nullptr);
    for (int i = 0; i < nums; i++)
        data[i] = u(e);
    if (this->data != nullptr) {
        XCalloc::xfree(this->data);
        this->data = nullptr;
    }
    this->shape = shape;
    this->data = data;
}

void Tensor::normalTensor(float mean, float sigma, vector<int>&& shape) {
    normalTensor(mean, sigma, shape);
}

void Tensor::onesTensor() {
    int nums = numel();
    for (int i = 0; i < nums; i++)
        data[i] = 1.0f;
}

void Tensor::onesTensor(vector<int>& shape) {
    int nums = 1;
    for (int& i : shape)
        nums *= i;
    if (shape.size() == 0)nums = 0;
    if (nums == numel())return onesTensor();
    float* data = (float*)XCalloc::xcalloc(nums, sizeof(float));
    assert(data != nullptr);
    for (int i = 0; i < nums; i++)
        data[i] = 1.0f;
    if (this->data != nullptr) {
        XCalloc::xfree(this->data);
        this->data = nullptr;
    }
    this->shape = shape;
    this->data = data;
}

void Tensor::onesTensor(vector<int>&& shape) {
    onesTensor(shape);
}

void Tensor::zerosTensor() {
    int nums = numel();
    data = (float*)memset(data, 0, nums * sizeof(float));
}

void Tensor::zerosTensor(vector<int>& shape) {
    int nums = 1;
    for (int& i : shape)
        nums *= i;
    if (shape.size() == 0)nums = 0;
    if (nums == numel())return zerosTensor();
    float* data = (float*)XCalloc::xcalloc(nums, sizeof(float));
    assert(data != nullptr);
    for (int i = 0; i < nums; i++)
        data[i] = 0.0f;
    if (this->data != nullptr) {
        XCalloc::xfree(this->data);
        this->data = nullptr;
    }
    this->shape = shape;
    this->data = data;
}

void Tensor::zerosTensor(vector<int>&& shape) {
    zerosTensor(shape);
}

float* Tensor::transpose_base(int dim1, int dim2, vector<int>& transpose_shape) {
    float* ans = (float*)XCalloc::xcalloc(numel(), sizeof(float));

    vector<int> dim_sum(shape.size(), 1);
    for (int i = shape.size() - 2; i >= 0; i--)
        dim_sum[i] = dim_sum[i + 1] * shape[i + 1];

    vector<int> transpose_dim_sum(shape.size(), 1);
    for (int i = shape.size() - 2; i >= 0; i--)
        transpose_dim_sum[i] = transpose_dim_sum[i + 1] * transpose_shape[i + 1];
#pragma omp parallel for
    for (int i = 0; i < numel(); i++) {
        int k = i;
        int transpose_pos = 0;
        for (int j = 0; j < shape.size(); j++) {
            if (j == dim1)transpose_pos += (k / dim_sum[j]) * transpose_dim_sum[dim2];
            else if (j == dim2) transpose_pos += (k / dim_sum[j]) * transpose_dim_sum[dim1];
            else transpose_pos += (k / dim_sum[j]) * transpose_dim_sum[j];
            k %= dim_sum[j];
        }
        ans[transpose_pos] = data[i];
    }
    return ans;
}

Tensor Tensor::transpose(int dim1, int dim2) {
    if (dim1 < 0)dim1 += dims();
    if (dim2 < 0)dim2 += dims();
    vector<int> transpose_shape(shape);
    if (dim1 >= shape.size() || dim2 >= shape.size()) {
        abort();
    }
    transpose_shape[dim1] = shape[dim2];
    transpose_shape[dim2] = shape[dim1];
    float* ans = transpose_base(dim1, dim2, transpose_shape);
    Tensor tensor;
    tensor.shape = transpose_shape;
    tensor.data = ans;
    return tensor;
}

void Tensor::clear() {
    shape.clear();
    if (data != nullptr) {
        XCalloc::xfree(data);
        data = nullptr;
    }
}

void Tensor::transpose_(int dim1, int dim2) {
    vector<int> transpose_shape(shape);
    if (dim1 >= shape.size() || dim2 >= shape.size()) {
        abort();
    }
    transpose_shape[dim1] = shape[dim2];
    transpose_shape[dim2] = shape[dim1];
    float* ans = transpose_base(dim1, dim2, transpose_shape);
    shape = transpose_shape;
    if (this->data != nullptr) {
        XCalloc::xfree(this->data);
        this->data = nullptr;
    }
    data = ans;
}

float Tensor::operator[](std::initializer_list<int>& pos_list) {
    if (pos_list.size() != shape.size()) {
        abort();
    }
    vector<int> dim_sum(shape.size(), 1);
    for (int i = shape.size() - 2; i >= 0; i--)
        dim_sum[i] = dim_sum[i + 1] * shape[i + 1];
    int pos = 0;

    auto begin = pos_list.begin();
    for (int i = 0; i < shape.size(); i++) {
        if (*(begin + i) >= shape[i]) {
            abort();
        }
        pos += (*(begin + i)) * dim_sum[i];
    }
    return data[pos];
}

float Tensor::operator[](std::initializer_list<int>&& pos_list) {
    return (*this)[pos_list];
}

void Tensor::matmul_(Tensor& other) {
    assert(dims() >= 2 && other.dims() >= 2);
    int W1 = size(-1), W2 = other.size(-1);
    int H1 = size(-2), H2 = other.size(-2);
    assert(W1 == H2);
    int length_max = max(dims(), other.dims()), length_min = min(dims(), other.dims());
    for (int i = -3; i >= -length_min; i--)assert(size(i) == other.size(i));

    vector<int> new_shape;
    for (int i = 0; i < length_max - 2; i++) {
        if (dims() > other.dims())new_shape.push_back(shape[i]);
        else new_shape.push_back(other.shape[i]);
    }
    new_shape.push_back(H1);
    new_shape.push_back(W2);

    int sum1_left = numel() / H1 / W1, sum2_left = other.numel() / H2 / W2;
    int max_left = max(sum1_left, sum2_left);
    int out_numel = max_left * H1 * W2;
    float* ans = (float*)XCalloc::xcalloc(out_numel, sizeof(float));
    assert(ans != nullptr);

    assert(max_left % sum1_left == 0 && max_left % sum2_left == 0);
    int size1 = H1 * W1, size2 = H2 * W2, out_per_size = H1 * W2;

    for (int i = 0; i < max_left; i++) {
        float* data1 = &data[(i % sum1_left) * size1];
        float* data2 = &other.data[(i % sum2_left) * size2];
        float* output = &ans[i * out_per_size];
        matrix_mm_fast(data1, H1, W1, data2, H2, W2, output);
    }

    if (data != nullptr) {
        XCalloc::xfree(data);
        data = nullptr;
    }
    this->shape = new_shape;
    data = ans;
}

void Tensor::add(Tensor& other) {
    int other_length = other.shape.size();
    int this_length = this->shape.size();
    assert(this_length >= other_length);
    for (int i = 0; i < other_length; i++)
        if (this->shape[this_length - i - 1] != other.shape[other_length - i - 1])
            abort();
    int size_gap = other.numel();
    int sum_add = numel() / size_gap;
    for (int i = 0; i < sum_add; i++) {
        int tmp = i * size_gap;
        for (int j = 0; j < size_gap; j++)
            this->data[tmp + j] += other.data[j];
    }
}

void Tensor::dot_(Tensor& other) {
    assert(dims() == other.dims());
    for (int i = 0; i < dims(); i++)
        assert(size(i) == other.size(i));
    for (int i = 0; i < numel(); i++)
        data[i] *= other.data[i];
}

Tensor Tensor::matmul(Tensor& other) {
    Tensor tmp = *this;
    tmp.matmul_(other);
    return std::move(tmp);
}

Tensor Tensor::matmul(Tensor&& other) {
    Tensor tmp = *this;
    tmp.matmul_(other);
    return std::move(tmp);
}

void matrix_mm(float* data1, int H1, int W1, float* data2, int H2, int W2, float* ans) {
    if (W1 != H2) {
        std::cout << "tensor::cpp-->the shape of two matrixes that are used to be multiply is not right!" << std::endl;
        abort();
    }
    assert(ans != nullptr);
#pragma omp parallel for
    for (int i = 0; i < H1; i++) {
        int data_pos = 0, data1_pos = 0, data2_pos = 0;
        float sum = 0.0f;
        for (int j = 0; j < W2; j++) {
            data_pos = i * W2 + j;
            for (int k = 0; k < H2; k++) {
                data2_pos = k * W2 + j;
                data1_pos = i * W1 + k;
                sum += data1[data1_pos] * data2[data2_pos];
            }
            ans[data_pos] = sum;
            sum = 0.0f;
        }
    }
}

void matrix_mm_fast(float* data1, int H1, int W1, float* data2, int H2, int W2, float* ans) {
    if (W1 != H2) {
        std::cout << "tensor::cpp-->the shape of two matrixes that are used to be multiply is not right!" << std::endl;
        abort();
    }
    assert(ans != nullptr);
    int i;
    if (CpuInfo::AVX512F()) {
#pragma omp parallel for private(i)
        for (i = 0; i < H1; i++) {
            int data1_pos = 0, data2_pos = 0, ans_pos = 0;
            for (int k = 0; k < H2; k++) {
                data1_pos = i * W1 + k;
                __m512 a0 = _mm512_set1_ps(data1[data1_pos]);
                __m512 b0, c0;
                __m512 result0;
                for (int j = 0; j < (W2 / 16) * 16; j += 16) {
                    b0 = _mm512_loadu_ps(&data2[k * W2 + j]);
                    c0 = _mm512_loadu_ps(&ans[i * W2 + j]);
                    result0 = _mm512_fmadd_ps(a0, b0, c0);
                    _mm512_storeu_ps(&ans[i * W2 + j], result0);
                }
                for (int j = (W2 / 16) * 16; j < W2; j++) {
                    data2_pos = k * W2 + j;
                    ans_pos = i * W2 + j;
                    ans[ans_pos] += data1[data1_pos] * data2[data2_pos];
                }
            }
        }
    }
    else if (CpuInfo::FMA()) {
#pragma omp parallel for private(i)
        for (i = 0; i < H1; i++) {
            int data1_pos = 0, data2_pos = 0, ans_pos = 0;
            for (int k = 0; k < H2; k++) {
                data1_pos = i * W1 + k;
                __m256 a0 = _mm256_set1_ps(data1[data1_pos]);
                __m256 b0, c0;
                __m256 result0;
                for (int j = 0; j < (W2 / 8) * 8; j += 8) {
                    b0 = _mm256_loadu_ps(&data2[k * W2 + j]);
                    c0 = _mm256_loadu_ps(&ans[i * W2 + j]);
                    result0 = _mm256_fmadd_ps(a0, b0, c0);
                    _mm256_storeu_ps(&ans[i * W2 + j], result0);
                }
                for (int j = (W2 / 8) * 8; j < W2; j++) {
                    data2_pos = k * W2 + j;
                    ans_pos = i * W2 + j;
                    ans[ans_pos] += data1[data1_pos] * data2[data2_pos];
                }
            }
        }
    }
    else if (CpuInfo::AVX()) {
#pragma omp parallel for private(i)
        for (i = 0; i < H1; i++) {
            int data1_pos = 0, data2_pos = 0, ans_pos = 0;
            for (int k = 0; k < H2; k++) {
                data1_pos = i * W1 + k;
                __m256 a0 = _mm256_set1_ps(data1[data1_pos]);
                __m256 b0, c0;
                __m256 result0;
                for (int j = 0; j < (W2 / 8) * 8; j += 8) {
                    b0 = _mm256_loadu_ps(&data2[k * W2 + j]);
                    c0 = _mm256_loadu_ps(&ans[i * W2 + j]);
                    result0 = _mm256_mul_ps(a0, b0);
                    result0 = _mm256_add_ps(result0, c0);
                    _mm256_storeu_ps(&ans[i * W2 + j], result0);
                }
                for (int j = (W2 / 8) * 8; j < W2; j++) {
                    data2_pos = k * W2 + j;
                    ans_pos = i * W2 + j;
                    ans[ans_pos] += data1[data1_pos] * data2[data2_pos];
                }
            }
        }
    }
    else if (CpuInfo::SSE()) {
#pragma omp parallel for private(i)
        for (i = 0; i < H1; i++) {
            int data1_pos = 0, data2_pos = 0, ans_pos = 0;
            for (int k = 0; k < H2; k++) {
                data1_pos = i * W1 + k;
                __m128 a0 = _mm_set1_ps(data1[data1_pos]);
                __m128 b0, c0;
                __m128 result0;
                for (int j = 0; j < (W2 / 4) * 4; j += 4) {
                    b0 = _mm_loadu_ps(&data2[k * W2 + j]);
                    c0 = _mm_loadu_ps(&ans[i * W2 + j]);
                    result0 = _mm_mul_ps(a0, b0);
                    result0 = _mm_add_ps(result0, c0);
                    _mm_storeu_ps(&ans[i * W2 + j], result0);
                }
                for (int j = (W2 / 4) * 4; j < W2; j++) {
                    data2_pos = k * W2 + j;
                    ans_pos = i * W2 + j;
                    ans[ans_pos] += data1[data1_pos] * data2[data2_pos];
                }
            }
        }
    }
    else {
        for (i = 0; i < H1; i++) {
            int data1_pos = 0, data2_pos = 0, ans_pos = 0;
            for (int k = 0; k < H2; k++) {
                data1_pos = i * W1 + k;
                for (int j = 0; j < W2; j++) {
                    data2_pos = k * W2 + j;
                    ans_pos = i * W2 + j;
                    ans[ans_pos] += data1[data1_pos] * data2[data2_pos];
                }
            }
        }
    }
}

Tensor matmul(Tensor& tensor1, Tensor& tensor2) {
    return std::move(tensor1.matmul(tensor2));
}

Tensor add(Tensor& tensor1, Tensor& tensor2) {
    Tensor tmp;
    if (tensor1.numel() < tensor2.numel()) {
        tmp = tensor2;
        tmp.add(tensor1);
    }
    else {
        tmp = tensor1;
        tmp.add(tensor2);
    }
    return std::move(tmp);
}

std::ostream& operator<<(std::ostream& cout, Tensor& tensor) {
    cout << "Shape:[";
    for (int i = 0; i < tensor.shape.size(); i++) {
        cout << tensor.shape[i];
        if (i != tensor.shape.size() - 1)
            cout << ",";
    }
    cout << "]\n";
    int sum = tensor.numel();
    int last_sum = tensor.size(-1);
    vector<int> dim_sum(tensor.shape.size(), last_sum);
    for (int i = tensor.shape.size() - 2; i >= 0; i--)
        dim_sum[i] = dim_sum[i + 1] * tensor.shape[i];
    std::stringstream ss = std::stringstream();
    int length = dim_sum.size();
    for (int i = 0; i < sum; i++) {
        int kuohao = 0;
        for (int& j : dim_sum)
            if ((i % j) == 0)kuohao++;
        if (kuohao > 0) {
            for (int j = 0; j < length - kuohao; j++)
                ss << " ";
            for (int j = 0; j < kuohao; j++)
                ss << "[";
        }
        kuohao = 0;
        ss << std::fixed << std::setprecision(8) << std::setw(12) << std::showpoint << tensor.data[i];
        for (int& j : dim_sum) {
            if (((i + 1) % j) == 0)kuohao++;
        }
        if (kuohao > 0) {
            for (int j = 0; j < kuohao; j++)
                ss << "]";
            if (i == sum - 1)ss << "\n";
            else ss << ",\n";
        }
        else ss << ",";
    }
    cout << ss.str();
    return cout;
}