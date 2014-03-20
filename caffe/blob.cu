// Copyright 2013 Yangqing Jia

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "blob.cuh"
#include "common.cuh"
#include "syncedmem.cuh"
#include "math_functions.cuh"


Blob::Blob(const int num, const int channels, const int height,
    const int width) {
  CHECK_GE(num, 0);
  CHECK_GE(channels, 0);
  CHECK_GE(height, 0);
  CHECK_GE(width, 0);
  num_ = num;
  channels_ = channels;
  height_ = height;
  width_ = width;
  count_ = num_ * channels_ * height_ * width_;
  CUDA_CHECK(cudaMalloc(&data_, count_ * sizeof(float)));
  CUDA_CHECK(cudaMemset(data_, 0, count_ * sizeof(float)));
  holder_ = true;
}

Blob::Blob(int num, int channels, int height, int width, int count, float* data)
  : num_(num), channels_(channels), height_(height), width_(width), count_(count), data_(NULL)
{
  CHECK_GE(num_, 0);
  CHECK_GE(channels_, 0);
  CHECK_GE(height_, 0);
  CHECK_GE(width_, 0);

  count_ = num_ * channels_ * height_ * width_;
  data_ = data;
  holder_ = false; //Blob donesn't hold the gpu data, it belongs to python
}

Blob::~Blob() {
  if (holder_) {
    CUDA_CHECK(cudaFree(data_));
  }
}

const float* Blob::gpu_data() const {
  CHECK(data_);
  return (const float*)data_;
}

float* Blob::mutable_gpu_data() {
  CHECK(data_);
  return reinterpret_cast<float*>(data_);
}
