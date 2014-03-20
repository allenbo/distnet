// Copyright 2013 Yangqing Jia

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "blob.cuh"
#include "common.cuh"
#include "syncedmem.cuh"
#include "math_functions.cuh"

void Blob::Reshape(const int num, const int channels, const int height,
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
  if (count_) {
    if (data_) {
      delete data_;
      delete diff_;
    }
    data_ = new SyncedMemory(count_ * sizeof(float));
    diff_ = new SyncedMemory(count_ * sizeof(float));
  } else {
    data_ = NULL;
    diff_ = NULL;
  }
}

Blob::Blob(const int num, const int channels, const int height,
    const int width) {
  Reshape(num, channels, height, width);
}

const float* Blob::cpu_data() const {
  CHECK(data_);
  return (const float*)data_->cpu_data();
}

const float* Blob::gpu_data() const {
  CHECK(data_);
  return (const float*)data_->gpu_data();
}

const float* Blob::cpu_diff() const {
  CHECK(diff_);
  return (const float*)diff_->cpu_data();
}

const float* Blob::gpu_diff() const {
  CHECK(diff_);
  return (const float*)diff_->gpu_data();
}

float* Blob::mutable_cpu_data() {
  CHECK(data_);
  return reinterpret_cast<float*>(data_->mutable_cpu_data());
}

float* Blob::mutable_gpu_data() {
  CHECK(data_);
  return reinterpret_cast<float*>(data_->mutable_gpu_data());
}

float* Blob::mutable_cpu_diff() {
  CHECK(diff_);
  return reinterpret_cast<float*>(diff_->mutable_cpu_data());
}

float* Blob::mutable_gpu_diff() {
  CHECK(diff_);
  return reinterpret_cast<float*>(diff_->mutable_gpu_data());
}

void Blob::Update() {
  // We will perform update based on where the data is located.
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
    // perform computation on GPU
    caffe_gpu_axpy(count_, float(-1),
        reinterpret_cast<const float*>(diff_->gpu_data()),
        reinterpret_cast<float*>(data_->mutable_gpu_data()));
    break;
  default:
    LOG(FATAL);
  }
}

void Blob::CopyFrom(const Blob& source, bool copy_diff, bool reshape) {
  if (num_ != source.num() || channels_ != source.channels() ||
      height_ != source.height() || width_ != source.width()) {
    if (reshape) {
      Reshape(source.num(), source.channels(), source.height(), source.width());
    } else {
      LOG(FATAL);
    }
  }
  if (copy_diff) {
    CUDA_CHECK(cudaMemcpy(diff_->mutable_gpu_data(), source.gpu_diff(),
        sizeof(float) * count_, cudaMemcpyDeviceToDevice));
  } else {
    CUDA_CHECK(cudaMemcpy(data_->mutable_gpu_data(), source.gpu_data(),
        sizeof(float) * count_, cudaMemcpyDeviceToDevice));
  }
}
