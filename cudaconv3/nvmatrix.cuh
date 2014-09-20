#ifndef NVMATRIX_H
#define NVMATRIX_H

#include "nvmatrix_kernels.cuh"
#include "nvmatrix_operators.cuh"
#include "pyassert.cuh"
#include <pthread.h>
#include <stdio.h>
#include "helper_cuda.h"
#include "cutil_inline.h"
#include <map>

#define CHECK_CUDA(msg)\
  cudaError_t err = cudaGetLastError();\
assert(cudaSuccess != err);

struct NVMatrix {
  int _numRows, _numCols, _stride, _numElements;
  float* _devData;
  cudaTextureObject_t _texObj;
  static pthread_mutex_t _streamMutex;
  static std::map<int,cudaStream_t> _defaultStreams;

  NVMatrix(float* gpuarray, int num_rows, int num_cols, int stride) :
    _devData(gpuarray), _numRows(num_rows), _numCols(num_cols) {
      _stride = stride;
      _numElements = _numRows * _numCols;
      _texObj = 0;
    }

  int getNumRows() const {
    return _numRows;
  }

  int getNumCols() const {
    return _numCols;
  }

  float* getDevData() {
    return _devData;
  }

  size_t getNumDataBytes() const {
    return size_t(_numRows * _numCols) * 4;
  }


  cudaTextureObject_t getTextureObject(){
   if (_texObj == 0) {
       assert(isContiguous());
       //size_t memFree, memTotal;

       struct cudaResourceDesc resDesc;
       memset(&resDesc, 0, sizeof(resDesc));
       resDesc.resType = cudaResourceTypeLinear;
       resDesc.res.linear.devPtr = getDevData();
       resDesc.res.linear.sizeInBytes = getNumDataBytes();
       resDesc.res.linear.desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
       struct cudaTextureDesc texDesc;
       memset(&texDesc, 0, sizeof(texDesc));
       checkCudaErrors(cudaCreateTextureObject(&_texObj, &resDesc, &texDesc, NULL));
   }
   assert(_texObj != 0);
   return _texObj;
  }
 
  static int getDeviceID() {
    int d;
    checkCudaErrors(cudaGetDevice(&d));
    return d;
  }
  static void setDeviceID(int d) {
    assert(d >= 0);
    checkCudaErrors(cudaSetDevice(d));
  }


  static cudaStream_t getDefaultStream() {
    return 0;
    return getDefaultStream(getDeviceID());
  }

  static cudaStream_t getDefaultStream(int deviceID) {
    if (deviceID >= 0) {
        pthread_mutex_lock(&_streamMutex);
        if (_defaultStreams.count(deviceID) == 0) {
            int oldDeviceID = getDeviceID();
            NVMatrix::setDeviceID(deviceID);
            checkCudaErrors(cudaStreamCreateWithFlags(&_defaultStreams[deviceID], cudaStreamNonBlocking));
            NVMatrix::setDeviceID(oldDeviceID);
        }
        cudaStream_t s = _defaultStreams[deviceID];
        pthread_mutex_unlock(&_streamMutex);
        return s;
    }
    return 0;
  }

  int getStride() const {
    return _stride;
  }

  int getLeadingDim() {
    return _numCols;
  }

  int getFollowingDim() {
    return _numRows;
  }
  int getNumElements() const {
    return getNumRows() * getNumCols();
  }
  bool isTrans() const {
    return false;
  }
  bool isSameDims(NVMatrix& other) const {
    return other.getNumRows() == getNumRows()
      && other.getNumCols() == getNumCols();
  }

  bool isContiguous() const {
    return true;
  }

  void resize(NVMatrix& like) {
    resize(like.getNumRows(), like.getNumCols());
  }

  void resize(int numRows, int numCols) const {
    if (!(_numRows == numRows && _numCols == numCols)) {
      throw Exception(
          StringPrintf("Cannot resize! (%d, %d) -> (%d, %d)", _numRows,
            _numCols, numRows, numCols), __FILE__, __LINE__);
    }
  }
  template<class Op>
    void apply(Op op) {
      NVMatrix& target = *this;
      int height = target.getNumRows();
      int width = target.getNumCols();
      dim3 blocks(std::min(NUM_BLOCKS_MAX, DIVUP(width, ELTWISE_THREADS_X)),
          std::min(NUM_BLOCKS_MAX, DIVUP(height, ELTWISE_THREADS_Y)));
      dim3 threads(ELTWISE_THREADS_X, ELTWISE_THREADS_Y);
      kEltwiseUnaryOp<Op><<<blocks, threads>>>(getDevData(), target.getDevData(), height, width, getStride(), target.getStride(), op);
      CHECK_CUDA("kEltwiseUnaryOp: Kernel execution failed");
    }

  template<class Agg, class UnaryOp, class BinaryOp>
  void _aggregate(int axis, NVMatrix& target, Agg agg, UnaryOp uop, BinaryOp bop, cudaStream_t stream) {
    assert(axis == 0 || axis == 1);
    assert(isContiguous()  && target.isContiguous());
    assert(&target != this);
    int width = isTrans() ? _numRows : _numCols;
    int height = isTrans() ? _numCols : _numRows;

    assert(width > 0);
    assert(height > 0);
    if((axis == 0 && !isTrans()) || (axis == 1 && isTrans())) { //col sum
        target.resize(!isTrans() ? 1 : _numRows, !isTrans() ? _numCols : 1);
//        int height = getFollowingDim();
          int numBlocks = DIVUP(width, NUM_SUM_COLS_THREADS_PER_BLOCK);
          assert(numBlocks * NUM_SUM_COLS_THREADS_PER_BLOCK >= width);
          assert(numBlocks < NUM_BLOCKS_MAX);
          kDumbAggCols<Agg, UnaryOp, BinaryOp><<<numBlocks,NUM_SUM_COLS_THREADS_PER_BLOCK, 0, stream>>>(getTextureObject(), target.getDevData(), width, height, agg, uop, bop);
          getLastCudaError("kDumbAggCols: Kernel execution failed");

        /*
        if ((height <= 2048 || width >= 4096)) {
            int numBlocks = DIVUP(width, NUM_SUM_COLS_THREADS_PER_BLOCK);
            assert(numBlocks * NUM_SUM_COLS_THREADS_PER_BLOCK >= width);
            assert(numBlocks < NUM_BLOCKS_MAX);
            kDumbAggCols<Agg, UnaryOp, BinaryOp><<<numBlocks,NUM_SUM_COLS_THREADS_PER_BLOCK, 0, stream>>>(getTextureObject(), target.getDevData(), width, height, agg, uop, bop);
            getLastCudaError("kDumbAggCols: Kernel execution failed");
        } else { // Specialize the case when we have very long columns and few of them
            const int sumLength = 128;
            NVMatrix tmp(DIVUP(height, sumLength), width);
            int numBlocksX = DIVUP(width, NUM_SUM_COLS_THREADS_PER_BLOCK);
            int numBlocksY = DIVUP(height, sumLength);
            dim3 blocks(numBlocksX, numBlocksY);
            dim3 threads(NUM_SUM_COLS_THREADS_PER_BLOCK);
            kAggCols<Agg, UnaryOp><<<blocks,threads, 0, stream>>>(getTextureObject(), tmp.getDevData(), width, height, sumLength, agg, uop);
            getLastCudaError("kAggCols: Kernel execution failed");

            int numBlocks = DIVUP(width, NUM_SUM_COLS_THREADS_PER_BLOCK);
            kDumbAggCols<Agg, NVMatrixOps::Identity, BinaryOp><<<numBlocks,NUM_SUM_COLS_THREADS_PER_BLOCK, 0, stream>>>(tmp.getTextureObject(), target.getDevData(), width, height, agg, NVMatrixOps::Identity(), bop);
            getLastCudaError("kDumbAggCols: Kernel execution failed");
        }
        */
    } else { // row sum
        target.resize(isTrans() ? 1 : _numRows, isTrans() ? _numCols : 1);
        if (width > 1) {
            if (height >= 16384) { // linear aggregation
                int numBlocksX = 1;
                int numBlocksY = DIVUP(height, AGG_SHORT_ROWS_THREADS_Y*AGG_SHORT_ROWS_LOOPS_Y);
                int numThreadsX = width <= 4 ? 4 : width <= 8 ? 8 : width <= 12 ? 12 : width <= 16 ? 16 : AGG_SHORT_ROWS_THREADS_X;
                int numThreadsY = AGG_SHORT_ROWS_THREADS_Y;
                while (numBlocksY > NUM_BLOCKS_MAX) {
                    numBlocksY = DIVUP(numBlocksY,2);
                    numBlocksX *= 2;
                }
                dim3 grid(numBlocksX, numBlocksY), threads(numThreadsX, numThreadsY);
                if(width <= 16) {
                    if(width <= 4) {
                        kAggShortRows<Agg, UnaryOp, BinaryOp, 1, 4><<<grid, threads, 0, stream>>>(getDevData(), target.getDevData(),width, height, agg, uop, bop);
                    } else if(width <= 8) {
                        kAggShortRows<Agg, UnaryOp, BinaryOp, 1, 8><<<grid, threads, 0, stream>>>(getDevData(), target.getDevData(),width, height, agg, uop, bop);
                    } else if(width <= 12) {
                        kAggShortRows<Agg, UnaryOp, BinaryOp, 1, 12><<<grid, threads, 0, stream>>>(getDevData(), target.getDevData(),width, height, agg, uop, bop);
                    } else {
                        kAggShortRows<Agg, UnaryOp, BinaryOp, 1, 16><<<grid, threads, 0, stream>>>(getDevData(), target.getDevData(),width, height, agg, uop, bop);
                    }
                } else if(width <= 32) {
                    kAggShortRows<Agg, UnaryOp, BinaryOp, 2, AGG_SHORT_ROWS_THREADS_X><<<grid, threads, 0, stream>>>(getDevData(), target.getDevData(),width, height, agg, uop, bop);
                } else if(width <= 48){
                    kAggShortRows<Agg, UnaryOp, BinaryOp, 3, AGG_SHORT_ROWS_THREADS_X><<<grid, threads, 0, stream>>>(getDevData(), target.getDevData(),width, height, agg, uop, bop);
                } else if(width <= 64){
                    kAggShortRows<Agg, UnaryOp, BinaryOp, 4, AGG_SHORT_ROWS_THREADS_X><<<grid, threads, 0, stream>>>(getDevData(), target.getDevData(),width, height, agg, uop, bop);
                } else {
                    kAggShortRows2<Agg, UnaryOp, BinaryOp><<<grid, threads, 0, stream>>>(getDevData(), target.getDevData(),width, height, agg, uop, bop);
                }
            } else {
                if (width >= 512) {
                    // NOTE: this is the only case which I bothered to try to optimize for Kepler
                    dim3 threads(AWR_NUM_THREADS);
                    dim3 blocks(1, height);
                    kAggRows_wholerow_nosync<<<blocks, threads, 0, stream>>>(getDevData(), target.getDevData(), width, height, agg, uop, bop);
                } else {

                    int numThreadsX = width <= 64 ? 32 : (width <= 128 ? 64 : (width <= 256 ? 128 : (width <= 512 ? 256 : 512)));
                    int numThreadsY = 1;
                    int numBlocksX = DIVUP(width, 2*numThreadsX);
                    int numBlocksY = std::min(height, NUM_BLOCKS_MAX);

                    dim3 grid(numBlocksX, numBlocksY), threads(numThreadsX, numThreadsY);
                    assert(numBlocksX <= NUM_BLOCKS_MAX);
                    assert(numBlocksY <= NUM_BLOCKS_MAX);

                    if(width <= 64) {
                        kAggRows<Agg, UnaryOp, BinaryOp, 32><<<grid, threads, 0, stream>>>(getDevData(), target.getDevData(),
                                                   width, height, target.getLeadingDim(), agg, uop, bop);
                    } else if(width <= 128) {
                        kAggRows<Agg, UnaryOp, BinaryOp, 64><<<grid, threads, 0, stream>>>(getDevData(), target.getDevData(),
                                                   width, height, target.getLeadingDim(), agg, uop, bop);
                    } else if(width <= 256) {
                        kAggRows<Agg, UnaryOp, BinaryOp, 128><<<grid, threads, 0, stream>>>(getDevData(), target.getDevData(),
                                                   width, height, target.getLeadingDim(), agg, uop, bop);
                    } else if(width <= 512) {
                        kAggRows<Agg, UnaryOp, BinaryOp, 256><<<grid, threads, 0, stream>>>(getDevData(), target.getDevData(),
                                                   width, height, target.getLeadingDim(), agg, uop, bop);
                    } else {
                        kAggRows<Agg, UnaryOp, BinaryOp, 512><<<grid, threads, 0, stream>>>(getDevData(), target.getDevData(),
                                                   width, height, target.getLeadingDim(), agg, uop, bop);
                    }

                    getLastCudaError("agg rows: Kernel execution failed");
                }
            }
        } else {
          assert(0);
        }
    }
  }

    template <class Op> void applyBinary(Op op, NVMatrix& b, NVMatrix& target, cudaStream_t stream) {
        assert(this->isSameDims(b));

        if (!target.isSameDims(*this)) {
            target.resize(*this);
        }

        if (getNumElements() > 0) {
            int height = target.getFollowingDim(), width = target.getLeadingDim();
            if (target.isTrans() == isTrans() && target.isTrans() == b.isTrans()) {
                if (!isContiguous() || !b.isContiguous() || !target.isContiguous()) {
                    dim3 blocks(std::min(128, DIVUP(width, ELTWISE_THREADS_X)),
                                std::min(128, DIVUP(height, ELTWISE_THREADS_Y)));
                    dim3 threads(ELTWISE_THREADS_X, ELTWISE_THREADS_Y);
                    kEltwiseBinaryOp<Op><<<blocks, threads, 0, stream>>>(getDevData(), b.getDevData(), target.getDevData(), height, width, getStride(),
                                                              b.getStride(), target.getStride(), op);
                } else {
                    dim3 threads = dim3(ELTWISE_FLAT_THREADS_X);
                    dim3 blocks = dim3(std::min(128, DIVUP(_numElements, ELTWISE_FLAT_THREADS_X)));
                    kEltwiseBinaryOpFlat<Op><<<blocks, threads, 0, stream>>>(getDevData(), b.getDevData(), target.getDevData(), _numElements, op);
                }
                getLastCudaError("kEltwiseBinaryOp: Kernel execution failed");
            } else {

                dim3 blocks(std::min(128, DIVUP(width, ELTWISE_THREADS_X)),
                            std::min(128, DIVUP(height, ELTWISE_THREADS_Y)));
                dim3 threads(ELTWISE_THREADS_X, ELTWISE_THREADS_Y);
                //  both x here since y divides x
                bool checkBounds = !(width % ELTWISE_THREADS_X == 0 && height % ELTWISE_THREADS_X == 0);
                if (target.isTrans() == isTrans() && target.isTrans() != b.isTrans()) {
                    if (checkBounds) {
                        kEltwiseBinaryOpTrans<Op,true,false,false><<<blocks, threads, 0, stream>>>(getDevData(), b.getDevData(), target.getDevData(), height, width,getStride(),
                                                                   b.getStride(), target.getStride(), op);
                    } else {
                        kEltwiseBinaryOpTrans<Op,false,false,false><<<blocks, threads, 0, stream>>>(getDevData(), b.getDevData(), target.getDevData(), height, width,getStride(),
                                                                   b.getStride(), target.getStride(), op);
                    }
                } else if (target.isTrans() != isTrans() && target.isTrans() != b.isTrans()) {
                    if (checkBounds) {
                        kEltwiseBinaryOpTrans<Op,true,true,false><<<blocks, threads, 0, stream>>>(getDevData(), b.getDevData(), target.getDevData(), height, width,getStride(),
                                                                   b.getStride(), target.getStride(), op);
                    } else {
                        kEltwiseBinaryOpTrans<Op,false,true,false><<<blocks, threads, 0, stream>>>(getDevData(), b.getDevData(), target.getDevData(), height, width,getStride(),
                                                                   b.getStride(), target.getStride(), op);
                    }
                } else if (target.isTrans() != isTrans() && target.isTrans() == b.isTrans()) {
                    if (checkBounds) {
                        kEltwiseBinaryOpTrans<Op,true,false,true><<<blocks, threads, 0, stream>>>(b.getDevData(), getDevData(), target.getDevData(), height, width,b.getStride(),
                                                                   getStride(), target.getStride(), op);
                    } else {
                        kEltwiseBinaryOpTrans<Op,false,false,true><<<blocks, threads, 0, stream>>>(b.getDevData(), getDevData(), target.getDevData(), height, width, b.getStride(),
                                                                   getStride(), target.getStride(), op);
                    }
                }
                getLastCudaError("kEltwiseBinaryOpTrans: Kernel execution failed");
            }
        }
    }



    template <class Op> void applyBinaryV(Op op, NVMatrix& vec, NVMatrix& target, cudaStream_t stream) {
        assert(&target != &vec); // for now
        if (isSameDims(vec)) {
            applyBinary(op, vec, target, stream);
            return;
        }
        assert(vec.getNumRows() == 1 || vec.getNumCols() == 1);
        assert(vec.getNumRows() == _numRows || vec.getNumCols() == _numCols);
        assert(vec.isContiguous());

        target.resize(*this); // target must be same orientation as me for now
        int width = getLeadingDim(); //_isTrans ? _numRows : _numCols;
        int height = getFollowingDim(); //_isTrans ? _numCols : _numRows;
        dim3 threads(ADD_VEC_THREADS_X, ADD_VEC_THREADS_Y);

        if ((vec.getNumRows() == _numRows && !isTrans()) || (vec.getNumCols() == _numCols && isTrans())) {
            dim3 blocks(std::min(512, DIVUP(width, ADD_VEC_THREADS_X)), std::min(NUM_BLOCKS_MAX, DIVUP(height, ADD_VEC_THREADS_Y)));
            kColVectorOp<Op><<<blocks, threads, 0, stream>>>(getDevData(), vec.getDevData(), target.getDevData(), width, height, getStride(), target.getStride(), op);
        } else {
            dim3 blocks(std::min(NUM_BLOCKS_MAX, DIVUP(width, ADD_VEC_THREADS_X)), std::min(NUM_BLOCKS_MAX, DIVUP(height, ADD_VEC_THREADS_Y)));
            kRowVectorOp<Op><<<blocks, threads, 0, stream>>>(getDevData(), vec.getDevData(), target.getDevData(), width, height, getStride(), target.getStride(), op);
        }
        getLastCudaError("Kernel execution failed");
    //    cudaThreadSynchronize();
    }


  void addVector(NVMatrix& vec) {
    applyBinaryV(NVMatrixBinaryOps::WeightedAdd(1, 1.0), vec, *this, getDefaultStream());
  }
};

#endif // NVMATRIX_H
