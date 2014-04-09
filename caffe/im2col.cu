// Copyright 2013 Yangqing Jia

// Modified by Justin Lin justin.lin@nyu.edu
#include <cmath>
#include <cstdlib>
#include <stdio.h>
#include <cstring>

#include "common.cuh"
#include "im2col.cuh"

__global__ void im2col_gpu_kernel(const int n, const float* data_im,
  const int height, const int width, const int ksize,
  const int stride, const int padding, const int height_col, const int width_col, float* data_col) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n) {
    int w_out = index % width_col;
    index /= width_col;
    int h_out = index % height_col;
    int channel_in = index / height_col;
    int channel_out = channel_in * ksize * ksize;
    int h_in = h_out * stride + padding;
    int w_in = w_out * stride + padding;
    data_col += (channel_out * height_col + h_out) * width_col + w_out;
    data_im += (channel_in * height + h_in) * width + w_in;
    for (int i = 0; i < ksize; ++i) {
      for (int j = 0; j < ksize; ++j) {
        if (i + h_in >= 0 && j + w_in >= 0 && i + h_in < height && j + w_in < width) {
          *data_col = data_im[i * width + j];
        }
        data_col += height_col * width_col;
      }
    }
  }
}

void im2col_gpu(const float* data_im, const int channels,
    const int height, const int width, const int ksize, const int stride, const int padding,
    float* data_col) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int height_col = (height - ksize - 2*padding + stride -1) / stride + 1;
  int width_col = (width - ksize - 2*padding + stride -1) / stride + 1;
  int num_kernels = channels * height_col * width_col;
  im2col_gpu_kernel<<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
    num_kernels, data_im, height, width, ksize, stride, padding, height_col, width_col,
    data_col);
  CUDA_POST_KERNEL_CHECK;
}


__global__ void col2im_gpu_kernel(const int n, const float* data_col,
  const int height, const int width, const int channels, const int ksize,
  const int stride, const int padding, const int height_col, const int width_col, float* data_im) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n) {
    float val = 0;
    int w = index % width - padding ;
    int h = (index / width) % height - padding;
    int c = index / (width * height);
    // compute the start and end of the output
    int w_col_start = (w < ksize) ? 0 : (w - ksize) / stride + 1;
    int w_col_end = min(w / stride + 1, width_col - 2 * padding);
    int h_col_start = (h < ksize) ? 0 : (h - ksize) / stride + 1;
    int h_col_end = min(h / stride + 1, height_col - 2 * padding);
    /*
    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
      for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
        // the col location: [c * width * height + h_out, w_out]
        int c_col = c * ksize * ksize + (h - h_col * stride) * ksize + (w - w_col * stride);
        val += data_col[(c_col * height_col + h_col) * width_col + w_col];
      }
    }
    */
    // equivalent implementation
    int offset = (c * ksize * ksize + h * ksize + w) * height_col * width_col;
    int coeff_h_col = (1 - stride * ksize * height_col) * width_col;
    int coeff_w_col = (1 - stride * height_col * width_col);
    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
      for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
        val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];
      }
    }
    data_im[index] = val;
  }
}

void col2im_gpu(const float* data_col, const int channels,
    const int height, const int width, const int ksize, const int stride, const int padding,
    float* data_im) {
  //CUDA_CHECK(cudaMemset(data_im, 0, sizeof(float) * height * width * channels));
  int height_col = (height - ksize - 2*padding + stride -1) / stride + 1;
  int width_col = (width - ksize - 2*padding + stride -1) / stride + 1;
  int num_kernels = channels * height * width;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  col2im_gpu_kernel<<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_col, height, width, channels, ksize, stride, padding,
      height_col, width_col, data_im);
  CUDA_POST_KERNEL_CHECK;
}
