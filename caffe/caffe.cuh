#ifndef CAFFE_CUH
#define CAFFE_CUH

#include "blob.cuh"

void convFilterActs(Blob& bottom, Blob& weight, Blob& top, int image_y, int output_y, int output_x, int padding, int stride, int color, int group);

void convImgActs(Blob& ingrad, Blob& weight, Blob& outgrad, int image_y, int image_x, int output_y, int padding, int stride, int color, int group);

void convWeightActs(Blob& input, Blob& ingrad, Blob& weight_grad, int image_y, int output_y, int output_x, int filter_size, int padding, int stride, int color, int num_group);

#endif
