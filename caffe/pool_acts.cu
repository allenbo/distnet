#include "blob.cuh"
#include "math_functions.cuh"
#include <assert.h>
#include <cfloat>

template <typename Dtype>
__global__ void MaxPoolForward(const int nthreads, const Dtype* bottom_data,
    const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int ksize, const int stride, Dtype* top_data) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < nthreads) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride;
    int hend = min(hstart + ksize, height);
    int wstart = pw * stride;
    int wend = min(wstart + ksize, width);
    Dtype maxval = -FLT_MAX;
    bottom_data += (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        maxval = max(maxval, bottom_data[h * width + w]);
      }
    }
    top_data[index] = maxval;
  }  // (if index < nthreads)
}

template <typename Dtype>
__global__ void MaxPoolBackward(const int nthreads, const Dtype* bottom_data,
    const Dtype* top_data, const Dtype* top_diff,
    const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int ksize, const int stride, Dtype* bottom_diff) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < nthreads) {
    // find out the local index
    // find out the local offset
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;
    int phstart = (h < ksize) ? 0 : (h - ksize) / stride + 1;
    int phend = min(h / stride + 1, pooled_height);
    int pwstart = (w < ksize) ? 0 : (w - ksize) / stride + 1;
    int pwend = min(w / stride + 1, pooled_width);
    Dtype gradient = 0;
    Dtype bottom_datum =
        bottom_data[((n * channels + c) * height + h) * width + w];
    top_data += (n * channels + c) * pooled_height * pooled_width;
    top_diff += (n * channels + c) * pooled_height * pooled_width;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        gradient += top_diff[ph * pooled_width + pw] *
            (bottom_datum == top_data[ph * pooled_width + pw]);
      }
    }
    bottom_diff[index] = gradient;
  }  // (if index < nthreads)
}


template <typename Dtype>
__global__ void AvgPoolForward(const int nthreads, const Dtype* bottom_data,
    const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int ksize, const int stride, Dtype* top_data) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < nthreads) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride;
    int hend = min(hstart + ksize, height);
    int wstart = pw * stride;
    int wend = min(wstart + ksize, width);
    Dtype aveval = 0;
    bottom_data += (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        aveval += bottom_data[h * width + w];
      }
    }
    top_data[index] = aveval / (hend - hstart) / (wend - wstart);
  }  // (if index < nthreads)
}

template <typename Dtype>
__global__ void AvgPoolBackward(const int nthreads, const Dtype* top_diff,
    const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int ksize, const int stride, Dtype* bottom_diff) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < nthreads) {
    // find out the local index
    // find out the local offset
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;
    int phstart = (h < ksize) ? 0 : (h - ksize) / stride + 1;
    int phend = min(h / stride + 1, pooled_height);
    int pwstart = (w < ksize) ? 0 : (w - ksize) / stride + 1;
    int pwend = min(w / stride + 1, pooled_width);
    Dtype gradient = 0;
    top_diff += (n * channels + c) * pooled_height * pooled_width;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        // figure out the pooling size
        int poolsize = (min(ph * stride + ksize, height) - ph * stride) *
            (min(pw * stride + ksize, width) - pw * stride);
        gradient += top_diff[ph * pooled_width + pw] / poolsize;
      }
    }
    bottom_diff[index] = gradient;
  }  // (if index < nthreads)
}


void convLocalMaxPool(Blob& input, Blob& output,
    int num_filter, int pool_size, int start, int stride,
    int input_y, int output_y, int output_x)
{
  const float* input_data = input.gpu_data();
  float* output_data = output.mutable_gpu_data();

  int input_x = input.width();
  
  assert(output.height() == output_y);
  assert(output.width() == output_x);
  assert(input.height() == input_y);
  assert(output.channels() == num_filter);
  assert(output.num() == input.num() && output.channels() == input.channels());

  int batch_size = input.num();
  int count = output.count();

  MaxPoolForward<float><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, input_data, num_filter, input_y, input_x, output_y, output_x,
      pool_size, stride, output_data);
  CUDA_POST_KERNEL_CHECK;
}



void convLocalAvgPool(Blob& input, Blob& output,
    int num_filter, int pool_size, int start, int stride,
    int input_y, int output_y, int output_x)
{
  const float* input_data = input.gpu_data();
  float* output_data = output.mutable_gpu_data();

  int input_x = input.width();
  
  assert(output.height() == output_y);
  assert(output.width() == output_x);
  assert(input.height() == input_y);
  assert(output.channels() == num_filter);
  assert(output.num() == input.num() && output.channels() == input.channels());

  int count = output.count();
  int batch_size = input.num();

  AvgPoolForward<float><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, input_data, num_filter, input_y, input_x, output_y, output_x,
      pool_size, stride, output_data);
  CUDA_POST_KERNEL_CHECK;
}



void convLocalMaxUndo(Blob& input, Blob& ingrad, Blob& output, Blob& outgrad,
    int pool_size, int start, int stride,
    int output_y, int output_x, int input_y)
{
  const float* ingrad_data = ingrad.gpu_data();
  const float* input_data = input.gpu_data();
  const float* output_data = output.gpu_data();
  float* outgrad_data = outgrad.mutable_gpu_data();

  const int input_x = input.width();

  assert(output.height() == output_y);
  assert(output.width() == output_x);
  assert(input.height() == input_y);

  int num_filter = output.channels();
  int count = input.count();
  int batch_size = input.num();
  
  MaxPoolBackward<float><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, input_data, output_data, ingrad_data, num_filter, input_y, input_x, 
      output_y, output_x, pool_size, stride, outgrad_data);
  CUDA_POST_KERNEL_CHECK;
}


void convLocalAvgUndo(Blob& ingrad, Blob& outgrad,
    int pool_size, int start, int stride,
    int output_y, int output_x, int input_y)
{
  const float* ingrad_data = ingrad.gpu_data();
  float* outgrad_data = outgrad.mutable_gpu_data();

  const int input_x = outgrad.width();

  assert(ingrad.height() == output_y);
  assert(ingrad.width() == output_x);
  assert(outgrad.height() == input_y);

  assert(ingrad.channels() == outgrad.channels());

  int num_filter = outgrad.channels();
  int count = outgrad.count();
  int batch_size = outgrad.num();

  AvgPoolBackward<float><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, ingrad_data, num_filter, input_y, input_x, output_y, output_x, pool_size, stride, outgrad_data);
  CUDA_POST_KERNEL_CHECK;
}
