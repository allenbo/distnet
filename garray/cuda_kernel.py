from distnet import util
from distnet.util import timer, divup, make_copy
from pycuda import gpuarray, driver
from pycuda.compiler import SourceModule
from pycuda.elementwise import ElementwiseKernel
from pycuda.gpuarray import GPUArray
from scikits.cuda import cublas
from time import time
import cPickle
import cudaconv
import numpy as np
import pycuda
import sys
from cudaconv import *
#cudaconv.init()

sgemm = None
def _initialize_cublas():
  global sgemm

  try:
    cublas.cublasInit()
    sgemm = cublas.cublasSgemm
  except AttributeError:
    handle = cublas.cublasCreate()
    def sgemm(*args):
      cublas.cublasSgemm(handle, *args)

#_initialize_cublas()
class CompiledSource(object):
  '''
  Compile a source string with PyCuda, caching the resulting module.
  '''
  def __init__(self, src, kernel_name):
    self.src = src
    self.kernel_name = kernel_name
    self.module = None
    self.kernel = None


  def __call__(self, *args, **kw):
    if self.module is None:
      util.log('Compiling... %s', self.kernel_name)
      self.module = SourceModule(self.src)
      self.kernel = self.module.get_function(self.kernel_name)

    self.kernel(*args, **kw)
    driver.Context.synchronize()


def I(i): return np.int32(i)
def F(f): return np.float32(f)

INTERNAL_SIZE = 256
_row_max_reduce_ = CompiledSource('''
    __global__
    void row_max_reduce(float* mat, float* vec, int leading, int rows, int cols) {
    const int INTERNAL_SIZE = 256;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    __shared__ float buffer[INTERNAL_SIZE];
    if(i < cols && i < INTERNAL_SIZE)
      buffer[i] = mat[i + j * leading];
    __syncthreads();

    int index = 1;
    if(cols > INTERNAL_SIZE) {
      if(threadIdx.x < INTERNAL_SIZE ) {
        int forwardInd = threadIdx.x + index * INTERNAL_SIZE;
        while(forwardInd < cols) {
          if (buffer[threadIdx.x] < mat[forwardInd + j* leading])
            buffer[threadIdx.x] = mat[forwardInd + j * leading];
          index ++;
          forwardInd = threadIdx.x + index * INTERNAL_SIZE;
        }
      }
    }
    __syncthreads();

    int total = INTERNAL_SIZE > cols? cols : INTERNAL_SIZE;
    while(total > 1) {
      int halfPoint = ((1+total) >> 1);
      if (threadIdx.x < halfPoint)  {
        if(threadIdx.x+halfPoint < total) {
          if(buffer[threadIdx.x] < buffer[threadIdx.x + halfPoint])
            buffer[threadIdx.x] = buffer[threadIdx.x + halfPoint];
        }
      }
      __syncthreads();
      total = ((1+total) >> 1);
    }
    __syncthreads();
    if(threadIdx.x == 0)
      vec[blockIdx.y] = buffer[0];
   }''', 'row_max_reduce')


_col_max_reduce_ = CompiledSource('''
    __global__
    void col_max_reduce(float* mat, float* vec, int leading, int rows, int cols) {
    const int INTERNAL_SIZE = 256;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    __shared__ float buffer[INTERNAL_SIZE];
    if(j < INTERNAL_SIZE && j < rows)
      buffer[j] = mat[i + j * leading];
    __syncthreads();

    int index = 1;
    if(rows > INTERNAL_SIZE) {
      if(threadIdx.y < INTERNAL_SIZE) {
        int forwardInd = threadIdx.y + index * INTERNAL_SIZE;
        while(forwardInd < rows) {
          if (buffer[threadIdx.y] < mat[i +forwardInd * leading])
            buffer[threadIdx.y] = mat[i  + forwardInd * leading];
          index ++;
          forwardInd = threadIdx.y + index * INTERNAL_SIZE;
        }
      }
    }
    __syncthreads();

    int total = INTERNAL_SIZE > rows ? rows : INTERNAL_SIZE;
    while(total > 1) {
      int halfPoint = ((1+total) >> 1);
      if (threadIdx.y < halfPoint)  {
        if(threadIdx.y+halfPoint < total) {
          if(buffer[threadIdx.y] < buffer[threadIdx.y + halfPoint])
            buffer[threadIdx.y] = buffer[threadIdx.y + halfPoint];
        }
      }
      __syncthreads();
      total = ((1+total) >> 1);
    }
    __syncthreads();
    if(threadIdx.y == 0)
      vec[i] = buffer[0];
   }
    ''', 'col_max_reduce')


_find_row_max_id_ = CompiledSource('''
    __global__
    void row_max_id(float* mat, float* vec, int leading, int rows, int cols) {
    const int INTERNAL_SIZE = 256;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    __shared__ float buffer[INTERNAL_SIZE];
    __shared__ int mind[INTERNAL_SIZE];
    if(i < INTERNAL_SIZE && i < cols){
      buffer[i] = mat[i + j * leading];
      mind[i] = threadIdx.x;
    }
    __syncthreads();

    int index = 1;
    if(cols > INTERNAL_SIZE)  {
      if(threadIdx.x < INTERNAL_SIZE) {
        int forwardInd = threadIdx.x + index * INTERNAL_SIZE;
        while(forwardInd < cols)  {
          if (buffer[threadIdx.x] < mat[forwardInd + j * leading]) {
            buffer[threadIdx.x] = mat[forwardInd + j * leading];
            mind[threadIdx.x] = forwardInd;
          }
          index ++;
          forwardInd = threadIdx.x + index * INTERNAL_SIZE;
        }
      }
    }
    __syncthreads();

    int total = INTERNAL_SIZE > cols ? cols : INTERNAL_SIZE;
    while(total > 1) {
      int halfPoint = ((1+total) >> 1);
      if (threadIdx.x < halfPoint)  {
        if(threadIdx.x+halfPoint < total) {
          if(buffer[threadIdx.x] < buffer[threadIdx.x + halfPoint]) {
            buffer[threadIdx.x] = buffer[threadIdx.x + halfPoint];
            mind[threadIdx.x] = mind[threadIdx.x + halfPoint];
          }
        }
      }
      __syncthreads();
      total = ((1+total) >> 1);
    }
    __syncthreads();
    if(threadIdx.x == 0)
      vec[blockIdx.y] = mind[0];
   }
    ''', 'row_max_id')


_find_col_max_id_ = CompiledSource('''
    __global__
    void col_max_id(float* mat, float* vec, int leading, int rows, int cols) {
    const int INTERNAL_SIZE = 256;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    __shared__ float buffer[INTERNAL_SIZE];
    __shared__ int mind[INTERNAL_SIZE];
    if( j < INTERNAL_SIZE && j < rows){
      buffer[j] = mat[i + j * leading];
      mind[j] = threadIdx.y;
     }
    __syncthreads();

    int index = 1;
    if(rows > INTERNAL_SIZE) {
      if(threadIdx.y < INTERNAL_SIZE ){
        int forwardInd = threadIdx.y + index * INTERNAL_SIZE;
        while(forwardInd < rows) {
          if (buffer[threadIdx.y] < mat[i + forwardInd * leading]) {
            buffer[threadIdx.y] = mat[i + forwardInd * leading];
            mind[threadIdx.y] = forwardInd;
          }
          index ++;
          forwardInd = threadIdx.y + index * INTERNAL_SIZE;
        }
      }
    }
    __syncthreads();

    int total = INTERNAL_SIZE > rows ? rows : INTERNAL_SIZE;
    while(total > 1) {
      int halfPoint = ((1+total) >> 1);
      if (threadIdx.y < halfPoint)  {
        if(threadIdx.y+halfPoint < total) {
          if(buffer[threadIdx.y] < buffer[threadIdx.y  + halfPoint]) {
            buffer[threadIdx.y] = buffer[threadIdx.y + halfPoint];
            mind[threadIdx.y] = mind[threadIdx.y + halfPoint];
          }
        }
      }
      __syncthreads();
      total = ((1+total) >> 1);
    }
    __syncthreads();
    if(threadIdx.y == 0)
      vec[i] = mind[0];
   }
    ''', 'col_max_id')


_add_vec_to_rows_ = CompiledSource('''
    __global__
    void add_vec_to_rows( float alpha, float* row, float beta, float* mat, float* dst,int leading, int rows, int cols) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      int j = blockIdx.y * blockDim.y + threadIdx.y;
      int index = i + j*leading;
      if ( i < cols   &&  j < rows)
        dst[index] = alpha* row[j] + beta * mat[index];
    }
    ''', 'add_vec_to_rows')


_add_vec_to_cols_ = CompiledSource('''
    __global__
    void add_vec_to_cols( float alpha, float* row, float beta, float* mat, float* dst,int leading, int rows, int cols) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      int j = blockIdx.y * blockDim.y + threadIdx.y;
      int index = i + j*leading;
      if ( i < cols   &&  j < rows)
        dst[index] = alpha* row[i] + beta * mat[index];
    }
    ''', 'add_vec_to_cols')


_div_vec_to_rows_ = CompiledSource('''
    __global__
    void div_vec_to_rows(float* row, float* mat, float* dst,int leading, int rows, int cols) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      int j = blockIdx.y * blockDim.y + threadIdx.y;
      int index = i + j*leading;
      if ( i < cols   &&  j < rows)
        dst[index] = __fdividef(mat[index], row[j]);
    }
    ''', 'div_vec_to_rows')

_div_vec_to_cols_ = CompiledSource('''
    __global__
    void div_vec_to_cols(float* row, float* mat, float* dst,int leading, int rows, int cols) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      int j = blockIdx.y * blockDim.y + threadIdx.y;
      int index = i + j*leading;
      if ( i < cols   &&  j < rows)
        dst[index] = __fdividef(mat[index], row[i]);
    }
    ''', 'div_vec_to_cols')

_add_row_sum_to_vec_ = CompiledSource(
  '''__global__ void add_row_sum(float* mat, float alpha, float* vec, float beta, int leading, int
  rows, int cols) {
    const int INTERNAL_SIZE = 256;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    __shared__ float buffer[INTERNAL_SIZE];
    if(i < cols)
      buffer[threadIdx.x] = mat[i + j * leading];
    __syncthreads();

    int total = INTERNAL_SIZE > cols ? cols : INTERNAL_SIZE;
    #pragma unroll
    while(total > 1) {
      int halfPoint = ((1+total) >> 1);
      if (threadIdx.x < halfPoint && i < cols)  {
        float temp = 0.0;
        if(threadIdx.x+halfPoint < total && i + halfPoint < cols) {
          temp = buffer[threadIdx.x + halfPoint];
        }
        buffer[threadIdx.x] += temp;
      }
      __syncthreads();
      total = ((1+total) >> 1);
    }
    __syncthreads();

    if(threadIdx.x == 0)
      vec[blockIdx.y * gridDim.x + blockIdx.x]  = alpha* vec[blockIdx.y * gridDim.x + blockIdx.x] + beta * buffer[0];
      //vec[j] = alpha*vec[j] + beta * buffer[0];
  }''', 'add_row_sum')


_add_col_sum_to_vec_ = CompiledSource('''
  __global__ void add_col_sum(float* mat, float alpha, float* vec, float beta, int leading, int
  rows, int cols) {
    const int INTERNAL_SIZE = 256;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    __shared__ float buffer[INTERNAL_SIZE];
    if(j < INTERNAL_SIZE && j < rows)
      buffer[j] = mat[i + j * cols];

    __syncthreads();

    int index = 1;
    if(rows > INTERNAL_SIZE) {
      if(threadIdx.y < INTERNAL_SIZE) {
        int forwardInd = threadIdx.y + index * INTERNAL_SIZE;
        while( forwardInd < rows) {
          buffer[threadIdx.y] += mat[i  + forwardInd * leading];
          index ++;
          forwardInd = threadIdx.y + index * INTERNAL_SIZE;
        }
      }
    }
    __syncthreads();

    int total = INTERNAL_SIZE > rows ? rows : INTERNAL_SIZE;
    while(total > 1) {
      int halfPoint = ((1+total) >> 1);
      if (threadIdx.y < halfPoint)  {
        float temp = 0.0;
        if(threadIdx.y+halfPoint < total) {
          temp = buffer[threadIdx.y + halfPoint];
        }
        buffer[threadIdx.y] += temp;
      }
      __syncthreads();
      total = ((1+total) >> 1);
    }
    __syncthreads();

    if(threadIdx.y == 0)
      vec[i]  = alpha* vec[i] + beta * buffer[0];
  }''', 'add_col_sum')

_same_reduce_ = CompiledSource('''
    __global__
    void same(float* tgt, float* vec, float* tmp) {
      int i = threadIdx.x;
      if( tgt[i] == vec[i] )
        tmp[i] = 1;
      else
        tmp[i] = 0;

    }
  ''', 'same')

_same_reduce_multiview_ = CompiledSource('''
    __global__
    void same(float* tgt, float* vec, float* tmp, float* ids, int num_view) {
      int id = threadIdx.x;
      int j;
      int i;
      float ret[10];
      int num_ret[10];
      int k = 0;
      for(i = 0; i < num_view; i ++ ) {
        for(j = 0; j < k; j ++ ) {
          if(vec[id + blockDim.x * i] == ret[j]) {
            num_ret[j] ++;
            break;
          }
        }
        if( j == k ) {
          num_ret[k] = 1;
          ret[k++] = vec[id+blockDim.x*i];
        }
      }
      int max = 0;
      float value = 0;
      for(i = 0; i < k; i ++ ) {
        if( max < num_ret[i] ) {
          max = num_ret[i];
          value = ret[i];
        }
      }
      ids[id] = value;

      if( tgt[id] == value ) {
        tmp[id] = 1;
      }else {
        tmp[id] = 0;
      }
    }
    ''', 'same')

_logreg_cost_row_reduce_ = CompiledSource('''
    __global__
    void log_reg_row(float* mat, float* label, float* cost, int leading){
      int i = threadIdx.x;
      int idx = i * leading + label[i];
      cost[i] = 0 - __logf(mat[idx]);
    }
    ''', 'log_reg_row')


_logreg_cost_col_reduce_ = CompiledSource('''
    __global__
    void log_reg_col(float* mat, float* label, float* cost, int leading){
      int i = threadIdx.x;
      int idx = i + label[i] * leading;
      cost[i] = 0 - __logf(mat[idx]);
    }
    ''', 'log_reg_col')


_softmax_bprop_ = CompiledSource(
      '''
      __global__
      void softmax_bprop_grad(float* mat, float* label, float* grad, int leading, int rows, int cols){
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;

        int idx= i + j * leading;
        if( i >= cols) return;
        if( j >= rows) return;

        if(j == label[i])
          grad[idx] = 1 - mat[idx];
        else
          grad[idx] = 0 - mat[idx];
      }
      ''', 'softmax_bprop_grad')

_relu_activate_ = CompiledSource('''
  __global__
  void relu_activate(float* input, float* output, float e,  int leading, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i >= cols) return ;
    if(j >= rows) return ;

    int idx = i + j * leading;

    output[idx] = fmaxf(input[idx], e);
  }''', 'relu_activate'
  )


_tanh_activate_ = CompiledSource('''
    __global__
    void tanh_activate(float* input, float *output, float a, float _n2b, int leading, int rows, int cols) {
     int i = blockIdx.x * blockDim.x + threadIdx.x;
      int j = blockIdx.y * blockDim.y + threadIdx.y;

      if(i >= cols) return ;
      if(j >= rows) return ;

      int idx = i + j * leading;

      output[idx] = a * (__fdividef(2.0f, 1.0f + __expf(input[idx]* _n2b)) - 1.0f);
    }''', 'tanh_activate'
    )

_relu_compute_grad_ = CompiledSource('''
  __global__
  void relu_compute_grad(float * grad, float * output, float* outGrad, float e, int leading, int rows, int
  cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i >= cols) return;
    if(j >= rows) return;

    int idx = i + j * leading;
    outGrad[idx] = grad[idx] * (output[idx] > e);
    //grad[idx] = grad[idx] * (output[idx] > e);
  }
  ''', 'relu_compute_grad')

_tanh_compute_grad_ = CompiledSource('''
  __global__
  void tanh_compute_grad(float * grad, float * output, float* outGrad, float a, float _n4ab,  int leading, int rows, int
  cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i >= cols) return;
    if(j >= rows) return;

    int idx = i + j * leading;
    float t = (1.0f - __fdividef(output[idx], a)) / 2.0f;
    outGrad[idx] = grad[idx] *_n4ab * (t * ( t - 1.0f));
    //grad[idx] = grad[idx] * (output[idx] > 0.0f);
  }
  ''', 'tanh_compute_grad')




_stride_copy_1_ = CompiledSource('''
    __global__
    void stride_copy_1(float* input, float* output, int start, int stride, int col, int reversed) {
      int i = threadIdx.x;
      int idx = i * stride + start;
      while ( i < col) {
        if (!reversed) {
          output[i] = input[idx];
        }else{
          input[idx] = output[i];
        }
        i += blockDim.x;
        idx = i * stride + start;
      }
    }''', 'stride_copy_1')


_stride_copy_2_ = CompiledSource ('''
    __global__
    void stride_copy_2(float* input, float* output,
      int start1, int stride1, int start2, int stride2,
      int col, int row, int ileading, int oleading, int reversed) {
    
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      int j = blockIdx.y * blockDim.y + threadIdx.y;

      if (i >= col || j >= row ) return;

      if (! reversed)
        output[i + j * oleading] = input[i * stride2 + start2 + ( j * stride1 + start1) * ileading];
      else
        input[i * stride2 + start2 + ( j * stride1 + start1) * ileading] = output[i + j * oleading];
    }''', 'stride_copy_2')


_stride_copy_3_image_block_ = CompiledSource('''
    __global__
    void stride_copy_3_image_block(float* input, float* output,
      int start1, int stride1, int start2, int stride2, int start3, int stride3,
      int ifleading, int isleading, int ofleading, int osleading, int reversed) {
      int z = blockIdx.x;
      int x = threadIdx.x;
      int y = threadIdx.y;

      int dind = z * ofleading + y * osleading + x;
      int sind = (z*stride1 + start1) * ifleading + (y * stride2 + start2) * isleading + (x * stride3+ start3);

      if (! reversed)
        output[dind] = input[sind];
      else
        input[sind] = output[dind];
    }''', 'stride_copy_3_image_block')



_stride_copy_3_channel_block_ = CompiledSource('''
    __global__
    void stride_copy_3_channel_block(float* input, float* output,
      int start1, int stride1, int start2, int stride2, int start3, int stride3, int col, int row, int channel,
      int ifleading, int isleading, int ofleading, int osleading, int reversed) {
      int x = blockIdx.x * blockDim.x + threadIdx.x;
      int y = blockIdx.y * blockDim.y + threadIdx.y;
      if ( x >= col || y >= row) return;

      int z = 0;
      int sidx, didx;
      while( z < channel) {
        didx = z * ofleading + y * osleading + x;
        sidx = (z*stride1 + start1) * ifleading + (y * stride2 + start2) * isleading + (x * stride3 + start3);
        if (! reversed)
          output[didx] = input[sidx];
        else
          input[sidx] = output[didx];
        z ++;
      }
    }''', 'stride_copy_3_channel_block')


_stride_copy_4 = ElementwiseKernel(
    '''float* input, float* output, 
    int start1, int start2, int start3, int start4,
    int step1, int step2, int step3, int step4,
    int sz1, int sz2, int sz3, int sz4,
    int istride1, int istride2, int istride3,
    int ostride1, int ostride2, int ostride3, int reversed
    ''',
    '''
    int idx[4];
    const int istride[] = { istride1, istride2, istride3, 1 };
    const int ostride[] = { ostride1, ostride2, ostride3, 1 };
    const int start[] = { start1, start2, start3, start4 };
    //const int sz[] = { sz1, sz2, sz3, sz4 };
    const int step[4] = { step1, step2, step3, step4 };

    int rest = i;
    int in_idx = 0;
    #pragma unroll 4
    for (int j = 0; j < 4; ++j) {
      idx[j] = rest / ostride[j];
      rest = rest % ostride[j];
      in_idx += istride[j] * (idx[j] * step[j] + start[j]);
    }


    if(!reversed) 
      output[i] = input[in_idx];
    else
      input[in_idx] = output[i];
    ''',
    name='stride_copy_4')

_stride_copy_sum = ElementwiseKernel(
    '''float* input, float* data,
    int start1, int start2, int start3, int start4,
    int step1, int step2, int step3, int step4,
    int sz1, int sz2, int sz3, int sz4,
    int istride1, int istride2, int istride3,
    int ostride1, int ostride2, int ostride3
    ''',
    '''
    int idx[4];
    const int istride[] = { istride1, istride2, istride3, 1 };
    const int ostride[] = { ostride1, ostride2, ostride3, 1 };
    const int start[] = { start1, start2, start3, start4 };
    //const int sz[] = { sz1, sz2, sz3, sz4 };
    const int step[4] = { step1, step2, step3, step4 };

    int rest = i;
    int in_idx = 0;
    #pragma unroll 4
    for (int j = 0; j < 4; ++j) {
      idx[j] = rest / ostride[j];
      rest = rest % ostride[j];
      in_idx += istride[j] * (idx[j] * step[j] + start[j]);
    }


    input[in_idx] += data[i];
    ''',
    name='stride_copy_sum')

_transpose_ = CompiledSource('''
  __global__
  void transpose(float * src, float* dst, int sleading, int dleading, int srows, int scols) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if(i >= scols) return ;
    if(j >= srows) return ;

    int sind = i + j * sleading;
    int dind = j + i * dleading;

    dst[dind] = src[sind];
  }''', 'transpose'
  )

_transpose_coalesced_ = CompiledSource('''
__global__ 
void transposeCoalesced(const float *idata, float *odata,int rows, int cols)
{
  const int TILE_DIM = 32;
  const int BLOCK_ROWS = 8;
  __shared__ float tile[TILE_DIM][TILE_DIM];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  if(x >= cols) return;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
     if(y + j >= rows) break;
     tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*cols + x];
  }

  __syncthreads();

  //x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
  //y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    if(y + j >= rows) continue;
     odata[x*rows + y + j] = tile[threadIdx.y+j][threadIdx.x];
  }
}
    ''', 'transposeCoalesced')



_transpose_diagonal_ = CompiledSource(
'''
__global__ void transposeDiagonal(float *idata,
 float *odata, int height,int width)
{
  const int TILE_DIM = 32;
  const int BLOCK_ROWS = 8;
  __shared__ float tile[TILE_DIM][TILE_DIM+1];
  int blockIdx_x, blockIdx_y;

  // diagonal reordering
  if (width == height) {
    blockIdx_y = blockIdx.x;
    blockIdx_x = (blockIdx.x+blockIdx.y)%gridDim.x;
  } else {
    int bid = blockIdx.x + gridDim.x*blockIdx.y;
    blockIdx_y = bid%gridDim.y;
    blockIdx_x = ((bid/gridDim.y)+blockIdx_y)%gridDim.x;
  }

  int xIndex = blockIdx_y*TILE_DIM + threadIdx.x;
  int yIndex = blockIdx_x*TILE_DIM + threadIdx.y;
  int index_out = xIndex + (yIndex)*height;

  xIndex = blockIdx_x*TILE_DIM + threadIdx.x;
  yIndex = blockIdx_y*TILE_DIM + threadIdx.y;
  int index_in = xIndex + (yIndex)*width;
  for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
    tile[threadIdx.y+i][threadIdx.x] = idata[index_in+i*width];
  }

  __syncthreads();

  for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
    odata[index_out+i*height] = tile[threadIdx.x][threadIdx.y+i];
  }
}
''', 'transposeDiagonal')

_matrix_add_ = CompiledSource('''
  __global__
  void matrix_add(float* src, float* v, float* dest, float alpha, float beta,  int leading, int
  rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i >= cols) return ;
    if(j >= rows) return ;

    int idx = i + j * leading;

    dest[idx] = src[idx] * alpha + v[idx] * beta;
  }''', 'matrix_add'
  )


_gpu_partial_copy_to_ = CompiledSource('''
    __global__
    void gpu_partial_copy_to(float* src, float* dest, int row_from, int row_to, int col_from, int
    col_to, int sleading, int dleading) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      int j = blockIdx.y * blockDim.y + threadIdx.y;

      if( i >= col_to - col_from) return;
      if( j >= row_to - row_from) return; 
      int sidx = i+col_from  + (j+ row_from) * sleading;
      int didx = i+ j  * dleading;

      dest[didx] = src[sidx];
    }''', 'gpu_partial_copy_to')

_bigger_than_scaler_ = CompiledSource('''
    __global__
    void bigger_than_scaler(float* src, float* dest, float scaler, int rows, int cols, int leading)
    {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      int j = blockIdx.y * blockDim.y + threadIdx.y;

      if (i >= cols) return ;
      if (j >= rows) return ;

      int idx = i + j * leading;

      dest[idx] = src[idx] >= scaler ? 1.0 : 0.0;
    }''', 'bigger_than_scaler')

_eltwise_exp_ = CompiledSource('''
    __global__
    void eltwise_exp(float* src, float* dest, int rows, int cols, int leading) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      int j = blockIdx.y * blockDim.y + threadIdx.y;

      if( i >= cols ) return ;
      if( j >= rows ) return ;

      int idx = i + j * leading;
      dest[idx] = __expf(src[idx]);
    }''', 'eltwise_exp')

_eltwise_mul_ = CompiledSource('''
    __global__
    void eltwise_mul(float* src, float* right, float* dest, int rows, int cols, int leading) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      int j = blockIdx.y * blockDim.y + threadIdx.y;

      if( i >= cols ) return ;
      if( j >= rows ) return ;

      int idx = i + j * leading;
      dest[idx] = src[idx] *right[idx];
    }''', 'eltwise_mul')

@util.timed_fn
def row_max_reduce(x, mat):
  '''
  Return the max of each row to a vec, ONLY work on small matrix
  Small means the column of the matrix is up to 1024
  and the rows, seams like a little big, can be 2048, but the upper bound has  not been tested
  '''
  mh, mw = mat.shape
  vh, vw = x.shape

  assert(vw == 1 and vh == mh or vh == 1 and vw == mh)

  grid = (1, mh)
  block = (mw, 1, 1)
  leading = mat.strides[0] / 4
  _row_max_reduce_(mat, x, I(leading), I(mh), I(mw), block=block, grid=grid)

@util.timed_fn
def col_max_reduce(x, mat):
  '''
  Return the max of each column to a vec, ONLY work on small matrix
  Small means the row of the matrix is up to 1024
  and the column, seams like a little big, can be 2048, but the upper bound has  not been tested
  '''
  mh, mw = mat.shape
  vh, vw = x.shape
  assert(vw == 1 and vh == mw or vh == 1 and vw == mw)

  grid = (mw, 1)
  block = (1, mh, 1)
  leading = mat.strides[0] / 4
  _col_max_reduce_(mat, x, I(leading), I(mh), I(mw), block=block, grid=grid)


@util.timed_fn
def find_row_max_id(x, mat):
  '''
  Return the id of max in each row to a vec(0-based), ONLY work on small matrix
  Small means the column of the matrix is up to 1024
  and the rows, seams like a little big, can be 2048, but the upper bound has  not been tested
  '''
  mh, mw = mat.shape
  vh, vw = x.shape
  assert(vw == 1 and vh == mh or vh == 1 and vw == mh), (x.shape, mat.shape)

  grid = (1, mh)
  block = (mw, 1, 1)
  leading = mat.strides[0] / 4
  _find_row_max_id_(mat, x, I(leading), I(mh), I(mw), block=block, grid=grid)
  


@util.timed_fn
def find_col_max_id(x, mat):
  '''
  Return the id of max in each column to a vec, ONLY work on small matrix
  Small means the row of the matrix is up to 1024
  and the column, seams like a little big, can be 2048, but the upper bound has  not been tested
  '''
  mh, mw = mat.shape
  vh, vw = x.shape
  assert(vw == 1 and vh == mw or vh == 1 and vw == mw)

  grid = (mw, 1)
  block = (1, mh, 1)
  leading = mat.strides[0] / 4

  _find_col_max_id_(mat, x, I(leading), I(mh), I(mw), block=block, grid=grid)
  



@util.timed_fn
def add_vec_to_rows(mat, vec, dest=None, alpha=1.0, beta=1.0):
  '''
  Add the element in vec to every element in mat in corresponding rows
  The function behaves exactly like mat + vec in numpy
  '''
  mh, mw = mat.shape
  vh, vw = vec.shape

  assert(vw == 1 and vh == mh or vh == 1 and vw == mh)

  if dest is None:
    dest = mat
  block = (32, 32, 1)
  grid = (divup(mw, 32), divup(mh, 32))
  leading = mat.strides[0] / 4
  _add_vec_to_rows_(F(alpha), vec, F(beta), mat, dest, I(leading), I(mh), I(mw), block=block, grid=grid)
  

@util.timed_fn
def add_vec_to_cols(mat, vec, dest=None, alpha=1.0, beta=1.0):
  '''
  Add the element in vec to every element in mat in corresponding cols
  The function behaves exactly like mat + vec in numpy
  '''
  mh, mw = mat.shape
  vh, vw = vec.shape

  assert(vw == 1 and vh == mw or vh == 1 and vw == mw)

  if not dest:
    dest = mat
  block = (32, 32, 1)
  grid = (divup(mw, 32), divup(mh, 32))
  leading = mat.strides[0] / 4
  _add_vec_to_cols_(F(alpha), vec, F(beta), mat, dest, I(leading), I(mh), I(mw), block=block, grid=grid)
  


@util.timed_fn
def div_vec_to_rows(mat, vec, dest=None):
  '''
  Divide the element in corresponding row of matrix by the element in the vec
  '''
  mh, mw = mat.shape
  vh, vw = vec.shape

  if not dest:
    dest = mat
  block = (32, 32, 1)
  grid = (divup(mw, 32), divup(mh, 32))
  leading = mat.strides[0] / 4
  _div_vec_to_rows_(vec, mat, dest, I(leading), I(mh), I(mw), block=block, grid=grid)
  



@util.timed_fn
def div_vec_to_cols(mat, vec, dest=None):
  '''
  Divide the element in corresponding column of matrix by the element in the vec
  '''
  mh, mw = mat.shape
  vh, vw = vec.shape

  if not dest:
    dest = mat
  block = (32, 32, 1)
  grid = (divup(mw , 32), divup(mh, 32))
  leading = mat.strides[0] / 4
  _div_vec_to_cols_(vec, mat, dest, I(leading), I(mh), I(mw), block=block, grid=grid)
  



@util.timed_fn
def add_row_sum_to_vec(vec, mat, alpha=1.0, beta=1.0):
  '''
  This function would sum up the element int a matrix row and store the result to
  the corresponding position of the vec
  Unlike other function that only provide small computation, this function raise the
  upper bound for the number of column to 2^16, actually it could be 2^20
  '''
  mh, mw = mat.shape
  vh, vw = vec.shape
  assert(vw == 1 and vh == mh or vh == 1 and vw == mh)
  if mw != 1:
    cudaconv.sum(mat, 1, vec)
  else:
    gpu_partial_copy_to(mat, vec, 0, mh, 0, 1)
  # if mat.shape[1] <= INTERNAL_SIZE:
  #  grid = (1, mh)
  #  block = (mw, 1,  1)
  #  leading = mat.strides[0] /4
  #  _add_row_sum_to_vec_(mat, F(alpha), vec, F(beta),I(leading), I(mh), I(mw), block = block, grid= grid)
  # else:
  #  block = (INTERNAL_SIZE, 1, 1)
  #  grid = (divup(mw, INTERNAL_SIZE), mh)
  #  #tmp  = gpuarray.to_gpu(np.zeros((mh, divup(mw, INTERNAL_SIZE)) ).astype(np.float32))
  #  tmp = gpuarray.zeros((mh, divup(mw, INTERNAL_SIZE)), dtype=np.float32)
  #  #print 'TOGPU', tmp.shape

  #  leading = mat.strides[0]/4
  #  _add_row_sum_to_vec_(mat, F(alpha), tmp, F(beta), I(leading), I(mh),I(mw), block = block, grid = grid)
  #  add_row_sum_to_vec(vec, tmp)
  


@util.timed_fn
def add_col_sum_to_vec(vec, mat, alpha=1.0, beta=1.0):
  '''
  This function would sum up the element int a matrix column and store the result to
  the corresponding position of the vec
  ONLY work on small matrix
  Small means the row of the matrix is up to 1024
  and the column, seams like a little big, can be 2048, but the upper bound has  not been tested
  '''
  mh, mw = mat.shape
  vh, vw = vec.shape
  assert(vw == 1 and vh == mw or vh == 1 and vw == mw)

  cudaconv.sum(mat, 0, vec)
  #grid = (mw, 1)
  #block = (1, mh, 1)
  #leading = mat.strides[0] / 4
  #_add_col_sum_to_vec_(mat, F(alpha), vec, F(beta), I(leading), I(mh), I(mw), block=block, grid=grid)
  


@util.timed_fn
def same_reduce(target, vec):
  '''
  Return the number of same values in the same offset of two vecs
  '''
  block = (target.size, 1, 1)
  grid = (1, 1)
  tmp = gpuarray.zeros_like(target)
  _same_reduce_(target, vec, tmp, block=block, grid=grid)
  tmp.shape = (1, tmp.size)
  res = gpuarray.to_gpu(np.zeros((1, 1)).astype(np.float32))
  add_row_sum_to_vec(res, tmp)
  
  return int(res.get()[0, 0])

@util.timed_fn
def same_reduce_multiview(target, vec, num_view):
  block = (target.size, 1, 1)
  grid = (1, 1)
  tmp = gpuarray.zeros_like(target)
  ids = gpuarray.zeros_like(target)
  _same_reduce_multiview_(target, vec, tmp, ids, I(num_view), block = block , grid = grid)
  tmp = tmp.reshape((1, tmp.size))
  res = gpuarray.to_gpu(np.zeros((1, 1)).astype(np.float32))
  add_row_sum_to_vec(res, tmp)

  return res.get()[0, 0]

@util.timed_fn
def logreg_cost_row_reduce(mat, label, cost):
  mh, mw = mat.shape
  vh, vw = label.shape
  assert(vh == 1 and vw == mh or vw == 1 and vh == mh)

  block = (mh, 1, 1)
  grid = (1, 1)
  _logreg_cost_row_reduce_(mat, label, cost, np.int32(mat.strides[0] / 4), block=block, grid=grid)
  


@util.timed_fn
def logreg_cost_col_reduce(mat, label, cost):
  mh, mw = mat.shape
  vh, vw = label.shape
  #assert(vh == 1 and vw == mw or vw == 1 and vh == mw)
  if (vh != 1 or vw != mw)  and (vw != 1 or vh != mw):
    util.log_info('%s ==> %s', mat.shape, label.shape)
    assert False

  block = (mw, 1, 1)
  grid = (1, 1)
  _logreg_cost_col_reduce_(mat, label, cost, np.int32(mat.strides[0] / 4), block=block, grid=grid)
  



@util.timed_fn
def softmax_bprop(mat, label, grad):
  mh, mw = mat.shape
  vh, vw = label.shape

  assert((vh == 1 and vw == mw) or (vw == 1 and vh == mw)), (vh, vw, mw)

  block = (32, 32, 1)
  grid = (divup(mw, 32), divup(mh, 32))
  _softmax_bprop_(mat, label, grad, I(mat.strides[0] / 4), I(mh), I(mw), block=block, grid=grid)
  

@util.timed_fn
def relu_activate(input, output, e):
  mh, mw = input.shape

  block = (32, 32, 1)
  grid = (divup(mw, 32), divup(mh, 32))
  leading = input.strides[0] / 4
  _relu_activate_(input, output, F(e), I(leading), I(mh), I(mw), block=block , grid=grid)
  


@util.timed_fn
def relu_compute_grad(grad, output, outGrad, e):
  mh, mw = grad.shape

  block = (32, 32, 1)
  grid = (divup(mw, 32), divup(mh, 32))
  leading = grad.strides[0] / 4
  _relu_compute_grad_(grad, output, outGrad, F(e), I(leading), I(mh), I(mw), block=block, grid=
      grid)
  

@util.timed_fn
def tanh_activate(input, output, a, b):
  mh, mw = input.shape

  block = (32, 32, 1)
  grid = (divup(mw, 32), divup(mh, 32))
  leading = input.strides[0] / 4
  _n2b = -2.0 * b
  _tanh_activate_(input, output, F(a), F(_n2b), I(leading), I(mh), I(mw), block=block , grid=grid)
  


@util.timed_fn
def tanh_compute_grad(grad, output, outGrad, a, b):
  mh, mw = output.shape

  block = (32, 32, 1)
  grid = (divup(mw, 32), divup(mh, 32))
  leading = output.strides[0] / 4
  _n4ab = -4.0 * a * b
  _tanh_compute_grad_(grad, output, outGrad, F(a), F(_n4ab), I(leading), I(mh), I(mw), block=block , grid=grid)
  


def stride_copy_1(input, output, slices):
  assert len(input.strides) == 1 and len(output.strides) == 1
  assert len(slices) == 1

  start, stop, stride = slices[0].indices(input.shape[0])
  if stride == 1:
    pycuda.driver.memcpy_dtod(output.gpudata,
        input.ptr + start * input.dtype.itemsize,
        (stop - start) * input.dtype.itemsize)
    return
  block = (128, 1, 1)
  grid = (1, 1)
  _stride_copy_1_(input, output, I(start),I(stride), I(output.size), I(0), block = block, grid = grid)


def stride_copy_2(input, output, slices):
  assert len(input.strides) ==2 and len(output.strides) == 2
  assert len(slices) == 2

  start1, stop1, stride1 = slices[0].indices(input.shape[0])
  start2, stop2, stride2 = slices[1].indices(input.shape[1])

  if stride1 == 1 and stride2 == 1:
    gpu_partial_copy_to(input, output, row_from = start1, row_to = stop1, col_from = start2, col_to = stop2)
  else:
    h, w = output.shape
    block = (32, 32, 1)
    grid = (divup(w, 32), divup(h, 32))
    ileading, oleading = input.strides[0] / 4, output.strides[0] / 4
    _stride_copy_2_(input, output,
        I(start1), I(stride1), I(start2), I(stride2),
        I(w), I(h), I(ileading), I(oleading), I(0),
        block = block, grid = grid)


def stride_copy_3(input, output, slices):
  assert len(input.strides) == 3 and len(output.strides) == 3
  assert len(slices) == 3

  start1, _, stride1 = slices[0].indices(input.shape[0])
  start2, _, stride2 = slices[1].indices(input.shape[1])
  start3, _, stride3 = slices[2].indices(input.shape[2])

  h, w = output.shape[1:]
  if h * w <= 1024:
    block = (w, h, 1)
    grid = (output.shape[0], 1)
    ifleading, isleading = input.strides[0]/4, input.strides[1]/4
    ofleading, osleading = output.strides[0]/4, output.strides[1]/4
    _stride_copy_3_image_block_(input, output,
        I(start1), I(stride1), I(start2), I(stride2), I(start3), I(stride3),
        I(ifleading), I(isleading), I(ofleading), I(osleading), I(0),
        block = block, grid = grid)
  else:
    block = (32, 32, 1)
    grid = (divup(w, 32), divup(h, 32))
    ifleading, isleading = input.strides[0]/4, input.strides[1]/4
    ofleading, osleading = output.strides[0]/4, output.strides[1]/4
    _stride_copy_3_channel_block_(input, output,
        I(start1), I(stride1), I(start2), I(stride2), I(start3), I(stride3),
        I(w), I(h), I(output.shape[0]),
        I(ifleading), I(isleading), I(ofleading), I(osleading), I(0), block = block , grid = grid)

def stride_copy_4(input, output, slices):
  assert len(input.strides) == 4 and len(output.strides) == 4
  assert len(slices) == 4

  start1, _, stride1 = slices[0].indices(input.shape[0])
  start2, _, stride2 = slices[1].indices(input.shape[1])
  start3, _, stride3 = slices[2].indices(input.shape[2])
  start4, _, stride4 = slices[3].indices(input.shape[3])

  sz1, sz2, sz3, sz4 = output.shape

  ifleading, isleading, itleading = [x / 4 for x in input.strides[:3]]
  ofleading, osleading, otleading = [x / 4 for x in output.strides[:3]]

  if start4 == 0 and stride4 == 1:
    copy = driver.Memcpy3D()
    copy.set_src_device(input.ptr)
    copy.set_dst_device(output.ptr)

    copy.width_in_bytes = 4 * sz4 * sz3
    copy.height = sz2
    copy.depth = sz1

    copy.src_pitch = 4 * input.shape[3] * input.shape[2]
    copy.dst_pitch = 4 * output.shape[3] * output.shape[2]

    copy.src_height = input.shape[1]
    copy.dst_height = output.shape[1]

    copy.src_z = start1
    copy.src_y = start2
    copy.src_x_in_bytes = 4 * (start3 * input.shape[3])

    copy.dst_z = 0
    copy.dst_y = 0
    copy.dst_x_in_bytes = 0


    copy()
    return
  _stride_copy_4(input, output,
      I(start1), I(start2), I(start3), I(start4),
      I(stride1), I(stride2), I(stride3), I(stride4),
      I(sz1), I(sz2), I(sz3), I(sz4),
      I(ifleading), I(isleading), I(itleading),
      I(ofleading), I(osleading), I(otleading),I(0),
      range=slice(0, np.prod(output.shape), 1))
  


def stride_copy(input, output, slices):
  if len(input.strides) == 1:
    stride_copy_1(input, output, slices)
  elif len(input.strides) == 2:
    stride_copy_2(input, output, slices)
  elif len(input.strides) == 3:
    stride_copy_3(input, output, slices)
  elif len(input.strides) == 4:
    stride_copy_4(input, output, slices)
  else:
    assert False
  return output


def stride_write_1(data, container, slices):
  assert len(data.strides) == 1 and len(container.strides) == 1
  assert len(slices) == 1

  start , _, stride = slices[0].indices(container.shape[0])

  if stride == 1:
    pycuda.driver.memcpy_dtod(container.ptr + start * container.dtype.itemsize,
        data.gpudata, data.nbytes)
    return
  block = (128, 1, 1)
  grid = (1, 1)
  _stride_copy_1_(container, data, I(start), I(stride), I(data.size), I(1), block = block, grid = grid)


def stride_write_2(data, container, slices):
  assert len(data.strides) == 2 and len(container.strides) == 2
  assert len(slices) == 2

  start1, stop1, stride1 = slices[0].indices(container.shape[0])
  start2, stop2, stride2 = slices[1].indices(container.shape[1])

  h, w = data.shape
  block = (32, 32, 1)
  grid = (divup(w, 32), divup(h, 32))
  ileading, oleading = container.strides[0] / 4, data.strides[0] / 4
  _stride_copy_2_(container, data,
      I(start1), I(stride1), I(start2), I(stride2),
      I(w), I(h), I(ileading), I(oleading), I(1),
      block = block, grid = grid)


def stride_write_3(data, container, slices):
  assert len(container.strides) == 3 and len(container.strides) == 3
  assert len(slices) == 3

  start1, _, stride1 = slices[0].indices(container.shape[0])
  start2, _, stride2 = slices[1].indices(container.shape[1])
  start3, _, stride3 = slices[2].indices(container.shape[2])

  h, w = data.shape[1:]
  if h * w <= 1024:
    block = (w, h, 1)
    grid = (data.shape[0], 1)
    ifleading, isleading = container.strides[0]/4, container.strides[1]/4
    ofleading, osleading = data.strides[0]/4, data.strides[1]/4
    _stride_copy_3_image_block_(container, data,
        I(start1), I(stride1), I(start2), I(stride2), I(start3), I(stride3),
        I(ifleading), I(isleading), I(ofleading), I(osleading), I(1),
        block = block, grid = grid)
  else:
    block = (32, 32, 1)
    grid = (divup(w, 32), divup(h, 32))
    ifleading, isleading = container.strides[0]/4, container.strides[1]/4
    ofleading, osleading = data.strides[0]/4, data.strides[1]/4
    _stride_copy_3_channel_block_(container, data,
        I(start1), I(stride1), I(start2), I(stride2), I(start3), I(stride3),
        I(w), I(h), I(data.shape[0]),
        I(ifleading), I(isleading), I(ofleading), I(osleading), I(1), block = block , grid = grid)

def stride_write_4(data, container, slices):
  assert len(container.strides) == 4 and len(data.strides) == 4
  assert len(slices) == 4

  start1, _, stride1 = slices[0].indices(container.shape[0])
  start2, _, stride2 = slices[1].indices(container.shape[1])
  start3, _, stride3 = slices[2].indices(container.shape[2])
  start4, _, stride4 = slices[3].indices(container.shape[3])


  sz1, sz2, sz3, sz4 = data.shape

  ifleading, isleading, itleading = [x / 4 for x in container.strides[:3]]
  ofleading, osleading, otleading = [x / 4 for x in data.strides[:3]]
  if start4 == 0 and stride4 == 1:
    copy = driver.Memcpy3D()
    copy.set_src_device(data.ptr)
    copy.set_dst_device(container.ptr)

    copy.width_in_bytes = 4 * sz4 * sz3
    copy.height = sz2
    copy.depth = sz1

    copy.src_pitch = data.strides[1]
    copy.dst_pitch = container.strides[1]

    copy.src_height = data.shape[1]
    copy.dst_height = container.shape[1]

    copy.src_z = 0
    copy.src_y = 0
    copy.src_x_in_bytes = 0

    copy.dst_z = start1
    copy.dst_y = start2
    copy.dst_x_in_bytes = 4 * (start3 * container.shape[3])

    copy()
    return

  _stride_copy_4(container, data,
      I(start1), I(start2), I(start3), I(start4),
      I(stride1), I(stride2), I(stride3), I(stride4),
      I(sz1), I(sz2), I(sz3), I(sz4),
      I(ifleading), I(isleading), I(itleading),
      I(ofleading), I(osleading), I(otleading),I(1),
      range=slice(0, np.prod(data.shape), 1))

def stride_write(data, container, slices):
  if len(data.strides) == 1:
    stride_write_1(data, container, slices)
  elif len(data.strides) == 2:
    stride_write_2(data, container, slices)
  elif len(data.strides) == 3:
    stride_write_3(data, container, slices)
  elif len(data.strides) == 4:
    stride_write_4(data, container, slices)
  else:
    assert False

def stride_write_sum(data, container, slices):
  assert len(container.strides) == 4 and len(data.strides) == 4
  assert len(slices) == 4

  start1, _, stride1 = slices[0].indices(container.shape[0])
  start2, _, stride2 = slices[1].indices(container.shape[1])
  start3, _, stride3 = slices[2].indices(container.shape[2])
  start4, _, stride4 = slices[3].indices(container.shape[3])


  sz1, sz2, sz3, sz4 = data.shape

  ifleading, isleading, itleading = [x / 4 for x in container.strides[:3]]
  ofleading, osleading, otleading = [x / 4 for x in data.strides[:3]]

  _stride_copy_sum(container, data,
      I(start1), I(start2), I(start3), I(start4),
      I(stride1), I(stride2), I(stride3), I(stride4),
      I(sz1), I(sz2), I(sz3), I(sz4),
      I(ifleading), I(isleading), I(itleading),
      I(ofleading), I(osleading), I(otleading),
      range=slice(0, np.prod(data.shape), 1))

@util.timed_fn
def gpu_copy_to(x, y):
  pycuda.driver.memcpy_dtod(y.gpudata, x.gpudata, x.nbytes)
  

@util.timed_fn
def gpu_partial_copy_to(x, y, row_from, row_to, col_from, col_to):
  mh, mw = x.shape
  row_to = min(row_to, mh)
  col_to = min(col_to, mw)
  r, c = row_to - row_from, col_to - col_from

  block = (32, 32, 1)
  grid = (divup(c, 32), divup(r, 32))
  sleading, dleading = x.strides[0] / 4, y.strides[0] / 4
  _gpu_partial_copy_to_(x, y, I(row_from), I(row_to), I(col_from), I(col_to), I(sleading), I(dleading), block=block, grid=grid)
'''
@util.lazyinit(_initialize_cublas)
@util.timed_fn
def matrixmult(x, y, atrans='t', btrans='t'):
  if isinstance(x, GPUArray):
    if atrans == 'n':
      shape = x.shape
      shape = (shape[1], shape[0])
      x = x.reshape(shape)

    if btrans == 'n':
      shape = y.shape
      shape = (shape[1], shape[0])
      y = y.reshape(shape)

    m = x.shape[0]
    n = y.shape[1]
    k = x.shape[1]

    assert k == y.shape[0], (x.shape, y.shape)

    xleading = x.shape[1] if atrans == 't' else x.shape[0]
    yleading = y.shape[1] if btrans == 't' else y.shape[0]

    result = GPUArray((n, m), dtype=x.dtype)
    sgemm(atrans, btrans, x.shape[0], y.shape[1], x.shape[1], 1.0, x.gpudata,xleading, y.gpudata,
        yleading, 0.0, result.gpudata, m)

    driver.Context.synchronize()
    return transpose(result)
  else:
    return np.dot(x, y)
'''
@util.lazyinit(_initialize_cublas)
@util.timed_fn
def matrixmult(x, y, dest = None):
  if isinstance(x, GPUArray):
    m = y.shape[1]
    n = x.shape[0]
    k = x.shape[1]

    assert k == y.shape[0], (x.shape, y.shape)

    if dest is None:
      result = GPUArray((n, m), dtype=x.dtype)
    else:
      result = dest
    sgemm('n', 'n', m, n, k, 1.0, y.gpudata, m, x.gpudata,
        k, 0.0, result.gpudata, m)

    driver.Context.synchronize()
    return result
  else:
    return np.dot(x, y)


@util.timed_fn
def transpose(mat, dst = None):
  mh, mw = mat.shape
  if dst is None:
    dst = gpuarray.empty((mw, mh), dtype=np.float32)

  block = (32, 8, 1)
  grid = (divup(mw, 32), divup(mh, 32))
  sleading = mat.strides[0] / 4
  dleading = dst.strides[0] / 4
  if mh % 32 == 0 and mw % 32 == 0:
    _transpose_diagonal_(mat, dst, I(mh), I(mw), block=block, grid=grid)
  else:
    _transpose_coalesced_(mat, dst, I(mh), I(mw), block=block, grid=grid)

  return dst

@util.timed_fn
def matrix_add(src, v, dest=None, alpha=1.0, beta=1.0):
  sh, sw = src.shape
  vh, vw = v.shape

  #assert sh == vh and sw == vw
  if sh != vh or sw != vw:
    assert False, '(%s, %s) + (%s, %s)' % (sh, sw, vh, vw)

  block = (32, 32, 1)
  grid = (divup(sw, 32), divup(sh, 32))
  leading = src.strides[0] / 4
  if dest is None:
    dest = src
  _matrix_add_(src, v, dest, F(alpha), F(beta), I(leading), I(sh), I(sw), 
               block=block , grid=grid)


@util.timed_fn
def bigger_than_scaler(src, scaler, dest=None):
  if dest is not None:
    assert dest.shape == src.shape
  else:
    dest = src

  mh, mw = src.shape

  block = (32, 32, 1)
  grid = (divup(mw, 32), divup(mh, 32))
  leading = src.strides[0] / 4
  _bigger_than_scaler_(src, dest, F(scaler), I(mh), I(mw), I(leading), block=block , grid=grid)

@util.timed_fn
def eltwise_exp(src, dest = None):
  if dest is None:
    dest = src
  mh, mw = src.shape

  block = (32, 32, 1)
  grid =  (divup(mw, 32), divup(mh, 32))
  leading = src.strides[0] / 4
  _eltwise_exp_(src, dest, I(mh), I(mw), I(leading), block = block, grid = grid)

@util.timed_fn
def eltwise_mul(src, right, dest = None):
  assert src.shape == right.shape
  if dest is None:
    dest = src
  mh, mw = src.shape

  block = (32, 32, 1)
  grid = (divup(mw, 32), divup(mh, 32))
  leading = src.strides[0] / 4
  _eltwise_mul_(src, right, dest, I(mh), I(mw), I(leading), block = block, grid = grid)
