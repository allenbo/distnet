// Copyright 2013 Yangqing Jia

#ifndef CAFFE_COMMON_HPP_
#define CAFFE_COMMON_HPP_

//#include <cublas_v2.h>
#include <cuda.h>
#include <curand.h>
// cuda driver types
#include <driver_types.h>
#include <stdio.h>

#define CHECK_GE(val1, val2) if ((val1) < (val2)) { \
  fprintf(stderr, #val1 " has to be greater than " #val2 " \n"); \
  exit(-1); \
  }

#define CHECK(val) if ((val) == 0) { \
  fprintf(stderr, #val " has to be nonzero\n"); \
  exit(-1); \
  }

#define CHECK_EQ(val1, val2) if ((val1) != (val2)) {\
  fprintf(stderr, #val1 " should be equal to " #val2 " \n"); \
  exit(-1); \
  }


#define LOG(x) do { fprintf(stderr, #x __FILE__ ":%d\n", __LINE__); \
  exit(-1); } while(0)

#define LOG_FATAL(fmt, ...) { fprintf(stderr, "[%s|%d]: FATAL ERROR " fmt, __FILE__, __LINE__, __VA_ARGS__); exit(-1);}

// various checks for different function calls.
#define CUDA_CHECK(condition) CHECK_EQ((condition), cudaSuccess)
#define CUBLAS_CHECK(condition) CHECK_EQ((condition), CUBLAS_STATUS_SUCCESS)
#define CURAND_CHECK(condition) CHECK_EQ((condition), CURAND_STATUS_SUCCESS)
#define VSL_CHECK(condition) CHECK_EQ((condition), VSL_STATUS_OK)

// After a kernel is executed, this will check the error and if there is one,
// exit loudly.
#define CUDA_POST_KERNEL_CHECK \
  cudaError_t ans = cudaGetLastError(); \
  if (cudaSuccess != ans) \
    LOG_FATAL("CUDA Error is %d:%s\n", ans, cudaGetErrorString(ans))

// Disable the copy and assignment operator for a class.
#define DISABLE_COPY_AND_ASSIGN(classname) \
private:\
  classname(const classname&);\
  classname& operator=(const classname&)

// Instantiate a class with float and double specifications.
#define INSTANTIATE_CLASS(classname) \
  template class classname<float>; \
  template class classname<double>

#define CAFFE_LOOP(i, count) \
  for(int i = threadIdx.x + blockIdx.x * blockDim.x; i < count; i += gridDim.x * blockDim.x) 

// We will use the boost shared_ptr instead of the new C++11 one mainly
// because cuda does not work (at least now) well with C++11 features.


// We will use 1024 threads per block, which requires cuda sm_2x or above.
//#if __CUDA_ARCH__ >= 200
//#warning __CUDA_ARCH__ is greater than 200
//#define CAFFE_CUDA_NUM_THREADS (1024)
//#else
//#define CAFFE_CUDA_NUM_THREADS (512)
//#endif

#define CAFFE_CUDA_NUM_THREADS (1024)
#define MAX_BLOCK (1 << 31 - 1)



inline int CAFFE_GET_BLOCKS(const int N) {
  int dim = (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
  return (dim < MAX_BLOCK)? dim : MAX_BLOCK;
  //return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
}

long cluster_seedgen(void);
#endif  // CAFFE_COMMON_HPP_
