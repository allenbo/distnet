#include "im2col.hpp"
#include "math_functions.hpp"
#include "blob.hpp"
#include <assert.h>

namespace caffe {

void convFilterActs(Blob& bottom, Blob& weight, Blob& top, int image_y, int output_y, int output_x, int padding, int stride, int color, int group) {
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

  Blob col_buffer_(0, 0, 0, 0);
  col_buffer_.Reshape(1, color * filter_size * filter_size, output_y, output_x);
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
        image_x, filter_size, stride, col_data);
    // Second, innerproduct with groups
    for (int g = 0; g < group; ++g) {
      caffe_gpu_gemm<float>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
        (float)1., weight_data + weight_offset * g, col_data + col_offset * g,
        (float)0., top_data + top.offset(n) + top_offset * g);
    }
  }
}


}
