// Copyright 2013 Yangqing Jia

#include <cmath>
#include <cstdlib>
#include <cstring>

#include "common.cuh"
#include "math_functions.cuh"

//template <>
//void caffe_gpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
//    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
//    const float alpha, const float* A, const float* B, const float beta,
//    float* C) {
//  // Note that cublas follows fortran order.
//  int lda = (TransA == CblasNoTrans) ? K : M;
//  int ldb = (TransB == CblasNoTrans) ? N : K;
//  cublasOperation_t cuTransA =
//      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
//  cublasOperation_t cuTransB =
//      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
//  CUBLAS_CHECK(cublasSgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
//      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
//}
//
//template <>
//void caffe_gpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
//    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
//    const double alpha, const double* A, const double* B, const double beta,
//    double* C) {
//  // Note that cublas follows fortran order.
//  int lda = (TransA == CblasNoTrans) ? K : M;
//  int ldb = (TransB == CblasNoTrans) ? N : K;
//  cublasOperation_t cuTransA =
//      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
//  cublasOperation_t cuTransB =
//      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
//  CUBLAS_CHECK(cublasDgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
//      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
//}
