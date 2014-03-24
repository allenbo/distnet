#ifndef CAFFE_CUH
#define CAFFE_CUH

#include "blob.cuh"

// Forward operation of conv layer
void convFilterActs(Blob& bottom, Blob& weight, Blob& top, int image_y, int output_y, int output_x, int padding, int stride, int color, int group);

// Backward operation of conv layer
void convImgActs(Blob& ingrad, Blob& weight, Blob& outgrad, int image_y, int image_x, int output_y, int padding, int stride, int color, int group);
void convWeightActs(Blob& input, Blob& ingrad, Blob& weight_grad, int image_y, int output_y, int output_x, int filter_size, int padding, int stride, int color, int num_group);

// Forward operation of max pooling layer
void convLocalMaxPool(Blob& bottom, Blob& top, int num_filter, int pool_size, int start, int stride, int image_y, int output_y, int output_x);

// Backward operation of max pooling layer
void convLocalMaxUndo(Blob& input, Blob& ingrad, Blob& output, Blob& outgrad, int pool_size, int start, int stride, int output_y, int output_x, int input_y);

// Forward operation of avg pooling layer
void convLocalAvgPool(Blob& bottom, Blob& top, int num_filter, int pool_size, int start, int stride, int image_y, int output_y, int output_x);

// Backward operation of max pooling layer
void convLocalAvgUndo(Blob& ingrad, Blob& outgrad, int pool_size, int start, int stride, int output_y, int output_x, int input_y);

// Forward operation of crmrnorm layer
void convResponseNormCrossMap(Blob& input, Blob& denoms, Blob& output, int num_filter, int norm_size, int input_y, float scaler, float pow, bool blocked);

// Backward operation of crmrnorm layer
void convResponseNormCrossMapUndo(Blob& ingrad, Blob& denoms, Blob& input, Blob& output, Blob& outgrad, int num_filter, int norm_size, int input_y, float scaler, float pow, bool blocked);

// Forward operation of rnorm layer
void convResponseNorm(Blob& input, Blob& denoms, Blob& output, int num_filter, int norm_size, int input_y, float scaler, float pow);

// Backward operation of rnorm layer
void convResponseNormUndo(Blob& ingrad, Blob& denoms, Blob& input, Blob& output, Blob& outgrad, int num_filter, int norm_size, int input_y, float scaler, float pow);


#endif
