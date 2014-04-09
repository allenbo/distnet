#include "blob.cuh"
#include "math_functions.cuh"
#include <assert.h>
#include <cfloat>
#include <iostream>

template <typename Dtype>
__global__ void CrossMapRNorm(const int nthreads, const Dtype* in, Dtype* out,
    const int num, const int channels, const int height,
    const int width, const int size, const Dtype alpha_over_size, const Dtype negative_beta,
    Dtype* scale) {
  CAFFE_LOOP(index, nthreads) {
    // find out the local offset
    int w = index % width;
    int h = (index / width) % height;
    int n = index / width / height;
    int offset = (n * channels * height + h) * width + w;
    int step = height * width;
    in += offset;
    scale += offset;
    int head = 0;
    int pre_pad = (size - 1) / 2;
    int post_pad = size - pre_pad - 1;
    Dtype accum_scale = 0;
    int index = 0;
    // fill the scale at [n, :, h, w]
    // accumulate values
    while (head < post_pad) {
      accum_scale += in[head * step] * in[head * step];
      ++head;
    }
    // until we reach size, nothing needs to be subtracted
    while (head < size) {
      index = (head - post_pad) * step;
      accum_scale += in[head * step] * in[head * step];
      scale[index] = 1. + accum_scale * alpha_over_size;
      out[index] = in[index] * pow(scale[index], negative_beta);
      ++head;
    }
    // both add and subtract
    while (head < channels) {
      accum_scale += in[head * step] * in[head * step];
      accum_scale -= in[(head - size) * step] * in[(head - size) * step];
      scale[(head - post_pad) * step] = 1. + accum_scale * alpha_over_size;
      out[index] = in[index] * pow(scale[index], negative_beta);
      ++head;
    }
    // subtract only
    while (head < channels + post_pad) {
      accum_scale -= in[(head - size) * step] * in[(head - size) * step];
      scale[(head - post_pad) * step] = 1. + accum_scale * alpha_over_size;
      out[index] = in[index] * pow(scale[index], negative_beta);
      ++head;
    }
  }
}

template <typename Dtype>
__global__ void CrossMapRNormComputeDiff(const int nthreads, const Dtype* bottom_data,
    const Dtype* top_data, const Dtype* scale, const Dtype* top_diff,
    const int num, const int channels, const int height, const int width, const int size,
    const Dtype negative_beta, const Dtype cache_ratio,
    Dtype* bottom_diff) {
  CAFFE_LOOP(index, nthreads) {
    // find out the local offset
    int w = index % width;
    int h = (index / width) % height;
    int n = index / width / height;
    int offset = (n * channels * height + h) * width + w;
    int step = height * width;
    bottom_data += offset;
    top_data += offset;
    scale += offset;
    top_diff += offset;
    bottom_diff += offset;
    int head = 0;
    int pre_pad = size - (size + 1) / 2;
    int post_pad = size - pre_pad - 1;
    Dtype accum_ratio = 0;
    // accumulate values
    while (head < post_pad) {
      accum_ratio += top_diff[head * step] * top_data[head * step] /
          scale[head * step];
      ++head;
    }
    // until we reach size, nothing needs to be subtracted
    while (head < size) {
      accum_ratio += top_diff[head * step] * top_data[head * step] /
          scale[head * step];
      bottom_diff[(head - post_pad) * step] = top_diff[(head - post_pad) * step]
          * pow(scale[(head - post_pad) * step], negative_beta) - cache_ratio *
          bottom_data[(head - post_pad) * step] * accum_ratio;
      ++head;
    }
    // both add and subtract
    while (head < channels) {
      accum_ratio += top_diff[head * step] * top_data[head * step] /
          scale[head * step];
      accum_ratio -= top_diff[(head - size) * step] *
          top_data[(head - size) * step] / scale[(head - size) * step];
      bottom_diff[(head - post_pad) * step] = top_diff[(head - post_pad) * step]
          * pow(scale[(head - post_pad) * step], negative_beta) - cache_ratio *
          bottom_data[(head - post_pad) * step] * accum_ratio;
      ++head;
    }
    // subtract only
    while (head < channels + post_pad) {
      accum_ratio -= top_diff[(head - size) * step] *
          top_data[(head - size) * step] / scale[(head - size) * step];
      bottom_diff[(head - post_pad) * step] = top_diff[(head - post_pad) * step]
          * pow(scale[(head - post_pad) * step], negative_beta) - cache_ratio *
          bottom_data[(head - post_pad) * step] * accum_ratio;
      ++head;
    }
  }
}

void convResponseNormCrossMap(Blob& input, Blob& denoms, Blob& output,
    int num_filter, int norm_size, int input_y, float scaler, float pow, bool blocked)
{
  const float* input_data = input.gpu_data();
  float* denoms_data = denoms.mutable_gpu_data();
  float* output_data = output.mutable_gpu_data();

  int n_threads = input.count() / num_filter;
  int input_x = input.width();
  int batch_size = input.num();
  
  CrossMapRNorm<float><<<CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS>>>(
      n_threads, input_data, output_data, batch_size,  num_filter, input_y, input_x,
      norm_size, scaler, -pow, denoms_data);
  CUDA_POST_KERNEL_CHECK;
}


void convResponseNormCrossMapUndo(Blob& ingrad, Blob& denoms, Blob& input, Blob& output, Blob& outgrad,
    int num_filter, int norm_size, int input_y, float scaler, float pow, bool blocked, float a, float b) 
{
  const float* input_data = input.gpu_data();
  const float* denoms_data = denoms.gpu_data();
  const float* output_data = output.gpu_data();
  const float* ingrad_data = ingrad.gpu_data();
  float* outgrad_data = outgrad.mutable_gpu_data();

  int batch_size = input.num();
  int input_x = input.width();
  int channel = input.channels();
  
  int n_threads = batch_size * input_y * input_x;
  
  CrossMapRNormComputeDiff<float><<<CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS>>>(
      n_threads, input_data, output_data, denoms_data, ingrad_data, batch_size, channel, input_y,
      input_x, norm_size, -pow, float(2. * scaler * pow), outgrad_data);

  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void RNorm(const int nthreads, const Dtype* bottom_data,
    const int channels, const int height, const int width,
    const int size, const Dtype alpha_over_size, const Dtype negative_beta, Dtype* scale_data, Dtype* top_data) {
  CAFFE_LOOP(index, nthreads) {
    const int pre_pad = (size - 1) / 2;
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;
    int hstart = max(0, h - pre_pad);
    int hend = min(h - pre_pad + size, height);
    int wstart = max(0, w - pre_pad);
    int wend = min(w - pre_pad + size, width);
    Dtype accum_scale = 0;
    bottom_data += (n * channels + c) * height * width;
    for (int hi = hstart; hi < hend; ++hi) {
      for (int wi = wstart; wi < wend; ++wi) {
        accum_scale += bottom_data[hi*width+wi] * bottom_data[hi*width+wi];
      }
    }
    scale_data[index] = pow(accum_scale + 1, negative_beta);
    top_data[index] = scale_data[index] * bottom_data[index];
  }  // (if index < nthreads)
}


template<typename Dtype>
__global__ void RNormComputeDiff(const int nthreads, const Dtype* bottom_data,
    const Dtype* top_data, const Dtype* scale_data, const Dtype* top_diff,
    const int channels, const int height, const int width,
    const int size, const Dtype negative_beta, const Dtype cache_ratio, Dtype* bottom_diff) {
  CAFFE_LOOP(index, nthreads) {
    const int pre_pad = (size - 1) / 2;
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;
    
    int hstart = max(0, h - pre_pad);
    int hend = min(h - pre_pad + size, height);
    int wstart = max(0, w - pre_pad);
    int wend = min(w - pre_pad + size, width);
  
    Dtype accum_ratio = 0;
    const Dtype* top_data_off = top_data + (n*channels + c) * height * width;
    const Dtype* scale_data_off = scale_data + (n*channels + c) * height * width;
    const Dtype* top_diff_off = top_diff + (n*channels + c) * height * width;

    for(int hi = hstart; hi < hend; hi ++) {
      for(int wi = wstart; wi < wend; wi ++){
        const int idx = hi * width + wi;
        accum_ratio += top_data_off[idx] * top_diff_off[idx] / scale_data_off[idx];
      }
    }

    bottom_diff[index] = top_diff[index] * pow(scale_data[index], negative_beta) 
      - cache_ratio * bottom_data[index] * accum_ratio;
  }// end if (index < nthreads)
}

void convResponseNorm(Blob& input, Blob& denoms, Blob& output,
    int num_filter, int norm_size, int input_y, float scaler, float pow)
{
  const float* input_data = input.gpu_data();
  float* denoms_data = denoms.mutable_gpu_data();
  float* output_data = output.mutable_gpu_data();

  int input_x = input.width();
  
  assert(input.height() == input_y);
  assert(output.channels() == num_filter);
  assert(output.num() == input.num() && output.channels() == input.channels());

  int batch_size = input.num();
  int count = output.count();

  RNorm<float><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, input_data, num_filter, input_y, input_x,
      norm_size, scaler, -pow, denoms_data, output_data);
  CUDA_POST_KERNEL_CHECK;
}

void convResponseNormUndo(Blob& ingrad, Blob& denoms, Blob& input, Blob& output, Blob& outgrad,
    int num_filter, int norm_size, int input_y, float scaler, float pow, float a, float b) 
{
  const float* input_data = input.gpu_data();
  const float* denoms_data = denoms.gpu_data();
  const float* output_data = output.gpu_data();
  const float* ingrad_data = ingrad.gpu_data();
  float* outgrad_data = outgrad.mutable_gpu_data();
 
  int count = input.count();
  int input_x = input.width();
  
  RNormComputeDiff<float><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, input_data, output_data, denoms_data, ingrad_data, num_filter, input_y, input_x,
      norm_size, -pow, float(2. * scaler * pow), outgrad_data);
  CUDA_POST_KERNEL_CHECK;
}
