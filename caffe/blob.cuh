// Copyright 2013 Yangqing Jia

#ifndef CAFFE_BLOB_HPP_
#define CAFFE_BLOB_HPP_

#include "common.cuh"
#include "pyassert.cuh"

class Blob {
  public:
    Blob(int num, int channels, int height, int width, int count, float* data);

    explicit Blob(const int num, const int channels, const int height,
        const int width);

    virtual ~Blob();

    inline int num() const { return num_; }
    inline int channels() const { return channels_; }
    inline int height() const { return height_; }
    inline int width() const { return width_; }

    inline int count() const {return count_; }

    inline int offset(const int n, const int c = 0, const int h = 0,
        const int w = 0) const {
      return ((n * channels_ + c) * height_ + h) * width_ + w;
    }

    const float* gpu_data() const;
    float* mutable_gpu_data();

  protected:
    float* data_;
    int num_;
    int channels_;
    int height_;
    int width_;
    int count_;

    bool holder_;

    DISABLE_COPY_AND_ASSIGN(Blob);
};  // class Blob
#endif
