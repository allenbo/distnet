#ifndef CAFFE_CUH
#define CAFFE_CUH

#include "blob.hpp"

void convFilterActs(Blob& bottom, Blob& weight, Blob& top, int image_y, int output_y,
                    int output_x, int padding, int stride, int color, int group);
#endif
