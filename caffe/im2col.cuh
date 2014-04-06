// Copyright 2013 Yangqing Jia

#ifndef _CAFFE_UTIL_IM2COL_HPP_
#define _CAFFE_UTIL_IM2COL_HPP_

void im2col_gpu(const float* data_im, const int channels,
    const int height, const int width, const int ksize, const int stride, const int paddin, 
    float* data_col);

void col2im_gpu(const float* data_col, const int channels,
    const int height, const int width, const int psize, const int stride, const int padding,
    float* data_im);

#endif  // CAFFE_UTIL_IM2COL_HPP_
