from pycuda import gpuarray
from pycuda.gpuarray import *
import numpy as np
from cuda_kernel import *
import cudaconv


def array(obj, dtype = np.float32):
  return to_gpu(obj).astype(dtype)

copy_to = gpu_copy_to

convolution = cudaconv.convFilterActs

def bconvolution(*args):
  cudaconv.convImgActs(*args, numGroups = 1)

def wconvolution(*args):
  cudaconv.convWeightActs(*args, numGroups = 1, partilSum = 0)

maxpool = cudaconv.convLocalMaxPool
maxundo = cudaconv.convLocalMaxUndo

avgpool = cudaconv.convLocalAvgPool
avgundo = cudaconv.convLocalAvgUndo

rnorm = cudaconv.convResponseNorm
def rnormundo(*args):
  cudaconv.convResponseNormUndo(*args, scaleTargets = 0.0, scaleOutput = 1.0)



old_add = GPUArray.__add__

def add(self, other):
  if len(other.shape) == 2:
    rst = zeros_like(self)
    copy_to(rst, self)
    if other.shape[0] == self.shape[0] and other.shape[1] == 1:
      return rst
    elif other.shape[1] == self.shape[1] and other.shape[0] == 1:
      add_vec_to_cols(rst, other)
    elif self.shape[0] == other.shape[0] and self.shape[1] == 1:
      add_row_sum_to_vec(rst, other)
    elif self.shape[1] == other.shape[1] and self.shape[0] == 1:
      ass_col_sum_to_vec(rst, other)
    else:
      assert False, 'Shape mismatch', self.shape, '+' , other.shape
    return rst
  return old_add(self, other)

GPUArray.__add__ = add
