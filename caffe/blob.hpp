// Copyright 2013 Yangqing Jia

#ifndef CAFFE_BLOB_HPP_
#define CAFFE_BLOB_HPP_

#include "syncedmem.hpp"

namespace caffe {

class Blob {
 public:
  Blob()
       : num_(0), channels_(0), height_(0), width_(0), count_(0), data_(),
       diff_() {
          if (num_ * channels_ * height_ * width_ != 0) {
            CHECK_GE(num_, 0);
            CHECK_GE(channels_, 0);
            CHECK_GE(height_, 0);
            CHECK_GE(width_, 0);

            count_ = num_ * channels_ * height_ * width_;
            data_ = new SyncedMemory(count_ * sizeof(float));
            diff_ = new SyncedMemory(count_ * sizeof(float));
          }
          else { 
            data_ = NULL;
            diff_ = NULL;
          }
       }
  explicit Blob(const int num, const int channels, const int height,
    const int width);
  virtual ~Blob() {
    if (data_) {
      delete data_;
      data_ = NULL;
      delete diff_;
      diff_ = NULL;
    }
  }
  void Reshape(const int num, const int height,
      const int width, const int channels);
  inline int num() const { return num_; }
  inline int channels() const { return channels_; }
  inline int height() const { return height_; }
  inline int width() const { return width_; }
  inline int count() const {return count_; }
  inline int offset(const int n, const int c = 0, const int h = 0,
      const int w = 0) const {
    return ((n * channels_ + c) * height_ + h) * width_ + w;
  }
  // Copy from source. If copy_diff is false, we copy the data; if copy_diff
  // is true, we copy the diff.
  void CopyFrom(const Blob& source, bool copy_diff = false,
      bool reshape = false);

  inline float data_at(const int n, const int c, const int h,
      const int w) const {
    return *(cpu_data() + offset(n, c, h, w));
  }

  inline float diff_at(const int n, const int c, const int h,
      const int w) const {
    return *(cpu_diff() + offset(n, c, h, w));
  }

  const float* cpu_data() const;
  const float* gpu_data() const;
  const float* cpu_diff() const;
  const float* gpu_diff() const;
  float* mutable_cpu_data();
  float* mutable_gpu_data();
  float* mutable_cpu_diff();
  float* mutable_gpu_diff();
  void Update();

 protected:
  SyncedMemory* data_;
  SyncedMemory* diff_;
  int num_;
  int channels_;
  int height_;
  int width_;
  int count_;

  DISABLE_COPY_AND_ASSIGN(Blob);
};  // class Blob

}  // namespace caffe

#endif  // CAFFE_BLOB_HPP_
