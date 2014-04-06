// Copyright 2013 Yangqing Jia

#ifndef CAFFE_UTIL_MATH_FUNCTIONS_H_
#define CAFFE_UTIL_MATH_FUNCTIONS_H_

//#include <cublas_v2.h>
#include "common.cuh"


// Decaf gemm provides a simpler interface to the gemm functions, with the
// limitation that the data has to be contiguous in memory.
// Decaf gpu gemm provides an interface that is almost the same as the cpu
// gemm function - following the c convention and calling the fortran-order
// gpu code under the hood.
//template <typename Dtype>
//void caffe_gpu_gemm(const CBLAS_TRANSPOSE TransA,
//    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
//    const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
//    Dtype* C);

#endif  // CAFFE_UTIL_MATH_FUNCTIONS_H_
