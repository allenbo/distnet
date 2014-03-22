#include "im2col.cuh"
#include "math_functions.cuh"
#include "blob.cuh"
#include <assert.h>
#include <iostream>

void convFilterActs(Blob& bottom, Blob& weight, Blob& top,
    int image_y, int output_y, int output_x,
    int padding, int stride, int color, int group) 
{
  const float* bottom_data = bottom.gpu_data();
  const float* weight_data = weight.gpu_data();
  float* top_data = top.mutable_gpu_data();

  int batch_size = bottom.num();
  int num_filter = weight.num();
  int filter_size = weight.height();

  int image_pixel = bottom.count() / (batch_size * color);
  int image_x = image_pixel / image_y;
  assert(bottom.channels() == color);
  assert(bottom.height() == image_y);
  assert(image_x * image_y == image_pixel);

  assert(padding <= 0);

  Blob col_buffer_(1, color * filter_size * filter_size, output_y, output_x);
  float* col_data = col_buffer_.mutable_gpu_data();

  int M_ = num_filter / group;
  int K_ = color * filter_size * filter_size / group;
  int N_ = output_y * output_x;

  int weight_offset = M_ * K_;
  int col_offset = K_ * N_;
  int top_offset = M_ * N_;
  for (int n = 0; n < batch_size; ++n) {
    // First, im2col
    im2col_gpu(bottom_data + bottom.offset(n), color, image_y,
        image_x, filter_size, stride, padding, col_data);
    // Second, innerproduct with groups
    for (int g = 0; g < group; ++g) {
      caffe_gpu_gemm<float>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
        (float)1., weight_data + weight_offset * g, col_data + col_offset * g,
        (float)0., top_data + top.offset(n) + top_offset * g);
    }
  }
}


void convImgActs(Blob& ingrad, Blob& weight, Blob& outgrad,
    int image_y, int image_x, int output_y,
    int padding, int stride, int color, int group)
{
  const float* ingrad_data = ingrad.gpu_data();
  const float* weight_data = weight.gpu_data();
  float* outgrad_data = outgrad.mutable_gpu_data();

  
  int batch_size = ingrad.num();
  int num_filter = ingrad.channels();
  int filter_size = weight.height();

  int output_pixel = ingrad.count() / (batch_size * num_filter);
  int output_x = output_pixel / output_y;
  assert(output_pixel == output_x * output_y);
  assert(outgrad.channels() == color);
  assert(padding <= 0);

  Blob col_buffer_(1, color * filter_size * filter_size, output_y, output_x);
  float* col_data = col_buffer_.mutable_gpu_data();

  int M_ = num_filter / group;
  int K_ = color * filter_size * filter_size / group;
  int N_ = output_y * output_x;

  int weight_offset = M_ * K_;
  int col_offset = K_ * N_;
  int ingrad_offset = M_ * N_;
  
  for(int n = 0; n < batch_size; n ++) {
    for (int g = 0; g < group; ++g) {
      caffe_gpu_gemm<float>(CblasTrans, CblasNoTrans, K_, N_, M_,
        (float)1., weight_data + weight_offset * g,
        ingrad_data + ingrad.offset(n) + ingrad_offset * g,
        (float)0., col_data + col_offset * g);
    }
    col2im_gpu(col_data, color, image_y,
        image_y, filter_size, stride, padding, outgrad_data + outgrad.offset(n));

  }
}
