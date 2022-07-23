#ifndef TENSOR_HPP
#define TENSOR_HPP
#include<vector>
#include<random>
#include<iostream>
#include<assert.h>
using std::vector;

int min(int, int);
int max(int, int);

void matrix_mm(float* data1, int H1, int W1, float* data2, int H2, int W2, float* ans);
void matrix_mm_fast(float* data1, int H1, int W1, float* data2, int H2, int W2, float* ans);
void matrix_mm(float* data1, int H1, int W1, float* data2, int H2, int W2, float* bias, float* ans);

class Tensor {
public:
    vector<int> shape;
    float* data;
public:
    int dims() const;
    int size(int dim) const;
    int numel() const;
    void reshape(vector<int>& new_shape);
    void reshape(vector<int>&& new_shape);
    ~Tensor();
    Tensor();
    Tensor(const Tensor&);
    Tensor(Tensor&&);
    Tensor(vector<int>& shape);
    Tensor(vector<int>&& shape);
    Tensor(vector<int>& shape, vector<float>& datas);
    Tensor(vector<int>& shape, vector<float>&& datas);
    Tensor(vector<int>& shape, std::initializer_list<float>& datas);
    Tensor(vector<int>& shape, std::initializer_list<float>&& datas);
    Tensor& operator=(const Tensor&);
    Tensor& operator=(Tensor&&);
public:
    void uniformTensor(float min, float max);
    void uniformTensor(float min, float max, vector<int>& shape);
    void uniformTensor(float min, float max, vector<int>&& shape);
    void normalTensor(float mean, float sigma);
    void normalTensor(float mean, float sigma, vector<int>& shape);
    void normalTensor(float mean, float sigma, vector<int>&& shape);
    void onesTensor();
    void onesTensor(vector<int>& shape);
    void onesTensor(vector<int>&& shape);
    void zerosTensor();
    void zerosTensor(vector<int>& shape);
    void zerosTensor(vector<int>&& shape);
    Tensor transpose(int dim1, int dim2);
    void transpose_(int dim1, int dim2);//
    float operator[](std::initializer_list<int>&);
    float operator[](std::initializer_list<int>&&);
    void matmul_(Tensor& other);
    void add(Tensor& other);
    void dot_(Tensor& other);
    Tensor matmul(Tensor& other);
    Tensor matmul(Tensor&& other);
    void clear();
private:
    float* transpose_base(int dim1, int dim2, vector<int>& transpose_shape);
};

Tensor matmul(Tensor& tensor1, Tensor& tensor2);
Tensor add(Tensor& tensor1, Tensor& tensor2);

std::ostream& operator<<(std::ostream& cout, Tensor& tensor);
#endif
