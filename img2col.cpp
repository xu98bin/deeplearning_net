#include "img2col.h"

void IMG2COL::img2col_cpu(float* input, int in_ch, int h, int w, int kernel_size, int stride, int padding, float* output) {
    int out_width = (w + 2 * padding - kernel_size) / stride + 1;
    int out_height = (h + 2 * padding - kernel_size) / stride + 1;
    int channel_size = h * w;
    int out_channels_length = out_width * out_height;
    for (int i = 0; i < out_height; i++) {
        for (int j = 0; j < out_width; j++) {
            for (int c = 0; c < in_ch; c++) {
                for (int kh = 0; kh < kernel_size; kh++) {
                    for (int kw = 0; kw < kernel_size; kw++) {
                        int src_pos_y = i * stride + kh;
                        int src_pos_x = j * stride + kw;
                        if (src_pos_y < padding || src_pos_y >= h + padding || src_pos_x < padding || src_pos_x >= w + padding)
                            continue;
                        int src_pos = (src_pos_y - padding) * w + src_pos_x - padding + c * channel_size;
                        int dst_pos = i * out_width + j + (c * kernel_size * kernel_size + kh * kernel_size + kw) * out_channels_length;
                        output[dst_pos] = input[src_pos];
                    }
                }
            }
        }
    }
}

void IMG2COL::col2img_cpu(float* delta_output, int in_ch, int h, int w, int kernel_size, int stride, int padding, float* delta_input) {
    int out_width = (w + 2 * padding - kernel_size) / stride + 1;
    int out_height = (h + 2 * padding - kernel_size) / stride + 1;
    int channel_size = h * w;
    int out_channels_length = out_width * out_height;
    for (int i = 0; i < out_height; i++) {
        for (int j = 0; j < out_width; j++) {
            for (int c = 0; c < in_ch; c++) {
                for (int kh = 0; kh < kernel_size; kh++) {
                    for (int kw = 0; kw < kernel_size; kw++) {
                        int src_pos_y = i * stride + kh;
                        int src_pos_x = j * stride + kw;
                        if (src_pos_y < padding || src_pos_y >= h + padding || src_pos_x < padding || src_pos_x >= w + padding)
                            continue;
                        int dst_pos = (src_pos_y - padding) * w + src_pos_x - padding + c * channel_size;
                        int src_pos = i * out_width + j + (c * kernel_size * kernel_size + kh * kernel_size + kw) * out_channels_length;
                        delta_output[dst_pos] += delta_input[src_pos];
                    }
                }
            }
        }
    }
}