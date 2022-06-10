#pragma once
#include <assert.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>
using std::cout;
using std::endl;

void transpose_matrix2D_InCol(float* data, int H, int W) {
    float* tmp = (float*)calloc(H * W, sizeof(float));
    tmp = (float*)memcpy(tmp, data, H * W * sizeof(float));
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            int tmp_pos = h * W + w;
            int data_pos = w * H + h;
            data[data_pos] = tmp[tmp_pos];
        }
    }
    free(tmp);
}

void transpose_matrix2D_InRow(float* data, int H, int W) {
    float* tmp = (float*)calloc(H * W, sizeof(float));
    tmp = (float*)memcpy(tmp, data, H * W * sizeof(float));
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            int data_pos = h * W + w;
            int tmp_pos = w * H + h;
            data[data_pos] = tmp[tmp_pos];
        }
    }
    free(tmp);
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

void matrix_dot_fast(float* data1/*changed*/, int H, int W, float* data2) {
    for (int h = 0; h < H; h ++) {
        __m256 a0, b0;
        for (int w = 0; w < (W / 8 * 8); w += 8) {
            a0 = _mm256_loadu_ps(&data1[h * W + w]);
            b0 = _mm256_loadu_ps(&data2[h * W + w]);
            b0 = _mm256_add_ps(a0, b0);
            _mm256_storeu_ps(&data1[h * W + w], b0);
        }
        for (int w = (W / 8 * 8); w < W; w++) {
            data1[h * W + w] += data2[h * W + w];
        }
    }
}

void matrix_dot(float* data1/*changed*/, int H, int W, float* data2) {
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            data1[h * W + w] += data2[h * W + w];
        }
    }
}