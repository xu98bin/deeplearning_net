#pragma once
#include <assert.h>
#include <iostream>
#include <stdlib.h>
#include <malloc.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>
using std::cout;
using std::endl;

void transpose_matrix2D_InRow(float* data, int H, int W) {
    //1024x1024 ---> usage time = 4133ms
    float* tmp = (float*)calloc(H * W, sizeof(float));
    tmp = (float*)memcpy(tmp, data, H * W * sizeof(float));
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            data[w * H + h] = tmp[h * W + w];
        }
    }
    free(tmp);
}

void transpose_matrix2D_InCol(float* data, int H, int W) {
    //1024x1024 ---> usage time = 2721ms
    float* tmp = (float*)calloc(H * W, sizeof(float));
    tmp = (float*)memcpy(tmp, data, H * W * sizeof(float));
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            data[h * W + w] = tmp[w * H + h];
        }
    }
    free(tmp);
}

void transpose_matrix2D_InCol_Split(float* data, int H, int W, int bits) {
    if (data == nullptr)return;
    float* tmp = (float*)calloc(H * W, sizeof(float));
    tmp = (float*)memcpy(tmp, data, H * W * sizeof(float));
    int block = bits / sizeof(data[0]);
    for (int h = 0; h < H / block; h++) {
        for (int w = 0; w < W / block; w++) {
            for (int i = 0; i < block; i++) {
                for (int j = 0; j < block; j++) {
                    data[(h * block + i) * W + w * block + j] = tmp[(w * block + j) * H + h * block + i];
                }
            }
        }
        for (int w = W / block * block; w < W; w++) {
            for (int i = 0; i < block; i++) {
                data[(h * block + i) * W + w] = tmp[w * H + h * block + i];
            }
        }
    }

    for (int h = H / block * block; h < H / block; h++) {
        for (int w = 0; w < W / block; w++) {
            for (int j = 0; j < block; j++) {
                data[h * W + w * block + j] = tmp[(w * block + j) * H + h];
            }
        }
        for (int w = W / block * block; w < W; w++) {
            data[h * W + w] = tmp[w * H + h];
        }
    }
    free(tmp);
}

void transpose_matrix2D_InCol_32x32(float* data, int H, int W) {
    //1024x1024x1024 ---> usage time = 1684ms  32
    transpose_matrix2D_InCol_Split(data, H, W, 32);
}

void transpose_matrix2D_InCol_64x64(float* data, int H, int W) {
    //1024x1024x1024 ---> usage time = 1697ms  64
    transpose_matrix2D_InCol_Split(data, H, W, 32);
}

void transpose_matrix2D_InCol_128x128(float* data, int H, int W) {
    //1024x1024x1024 ---> usage time = 1949ms
    transpose_matrix2D_InCol_Split(data, H, W, 32);
}

void matrix_mm_slow(float* data1, int H1, int W1, float* data2, int H2, int W2, float* ans) {
    //[H1,W1]X[H2,W2] = [H1,W2],[W1=H2]
    if (W1 != H2) {
        std::cout << "tensor::cpp-->the shape of two matrixes that are used to be multiply is not right!" << std::endl;
        abort();
    }
    assert(ans != nullptr);
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

void matrix_mm(float* data1, int H1, int W1, float* data2, int H2, int W2, float* ans) {
    if (W1 != H2) {
        std::cout << "tensor::cpp-->the shape of two matrixes that are used to be multiply is not right!" << std::endl;
        abort();
    }
    assert(ans != nullptr);
    for (int i = 0; i < H1; i++) {
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

void matrix_mm_fast(float* data1, int H1, int W1, float* data2, int H2, int W2, float* ans) {
    if (W1 != H2) {
        std::cout << "tensor::cpp-->the shape of two matrixes that are used to be multiply is not right!" << std::endl;
        abort();
    }
    assert(ans != nullptr);
#pragma omp parallel for
    for (int i = 0; i < H1; i++) {
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

void matrix_dot_fast(float* data1/*changed*/, int nums, float* data2) {
#pragma omp parallel for
    for (int i = 0; i < nums / 8 * 8; i += 8) {
        __m256 a0, b0;
        a0 = _mm256_load_ps(&data1[i]);
        b0 = _mm256_load_ps(&data2[i]);
        b0 = _mm256_mul_ps(a0, b0);
        _mm256_store_ps(&data1[i], b0);
    }

    for (int i = (nums / 8 * 8); i < nums; i++) {
        data1[i] *= data2[i];
    }
}

void matrix_dot(float* data1/*changed*/, int nums, float* data2) {
#pragma omp parallel for
    for (int i = 0; i < nums; i++) {
        data1[i] *= data2[i];
    }
}

void matrix_add(float* data1/*changed*/, int nums, float* data2) {
    for (int i = 0; i < nums / 8 * 8; i += 8) {
        __m256 a0, b0;
        a0 = _mm256_load_ps(&data1[i]);
        b0 = _mm256_load_ps(&data2[i]);
        b0 = _mm256_add_ps(a0, b0);
        _mm256_store_ps(&data1[i], b0);
    }

    for (int i = (nums / 8 * 8); i < nums; i++) {
        data1[i] *= data2[i];
    }
}