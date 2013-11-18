from pycuda import gpuarray, driver
from pycuda.gpuarray import GPUArray, to_gpu, zeros, zeros_like, empty
import numpy as np
from cuda_kernel import *

cudaconv.init()

old_init = GPUArray.__init__
def new_init(*args, **kw):
  driver.Context.synchronize()
  return old_init(*args, **kw)
GPUArray.__init__ = new_init


def reshape_last(input):
  shape = input.shape
  row = int(np.prod(shape[:-1]))
  col = shape[-1]
  return input.reshape((row, col))


def reshape_first(input):
  shape = input.shape
  row = shape[0]
  col = int(np.prod(shape[1:]))
  return input.reshape((row, col))


def array(obj, dtype = np.float32, to2dim = False):
  if len(obj.shape) != 2 and len(obj.shape) != 1 and to2dim:
    obj = reshape_last(obj)
  return to_gpu(obj).astype(dtype)


def get_seed():
  import time
  return int(time.time())



copy_to = gpu_copy_to
partial_copy_to = gpu_partial_copy_to


def partial_copy(input, f, t):
  shape = list(input.shape)
  shape[-1] = t - f
  data = zeros(tuple(shape), dtype = np.float32)
  partial_copy_to(input, data, 0, shape[0], f, t)
  return data


convolution = cudaconv.convFilterActs

def bconvolution(*args):
  args = args[1:] + (1,)
  cudaconv.convImgActs(*args)

def wconvolution(*args):
  args = args + (1, 0)
  cudaconv.convWeightActs(*args)

maxpool = cudaconv.convLocalMaxPool
maxundo = cudaconv.convLocalMaxUndo

avgpool = cudaconv.convLocalAvgPool
def avgundo(*args):
  args = args[1:]
  cudaconv.convLocalAvgUndo(*args)

rnorm = cudaconv.convResponseNorm
def rnormundo(*args):
  args = args + (0.0, 1.0)
  cudaconv.convResponseNormUndo(*args)

rnormcrossmap = cudaconv.convResponseNormCrossMap
def rnormcrossmapundo(*args):
  args = args +  (0.0, 1.0)
  cudaconv.convResponseNormCrossMapUndo(*args)


old_add = GPUArray.__add__
def newadd(self, other):
  if other.shape == self.shape:
    return old_add(self, other)
  if len(other.shape) == 2:
    rst = zeros_like(self)
    copy_to(self, rst)
    if other.shape[0] == self.shape[0] and other.shape[1] == 1:
      add_vec_to_rows(rst, other)
    elif other.shape[1] == self.shape[1] and other.shape[0] == 1:
      add_vec_to_cols(rst, other)
    elif self.shape[0] == other.shape[0] and self.shape[1] == 1:
      add_row_sum_to_vec(rst, other)
    elif self.shape[1] == other.shape[1] and self.shape[0] == 1:
      ass_col_sum_to_vec(rst, other)
    else:
      assert False, 'Shape mismatch' + str(self.shape) + '+' + str(other.shape)
    return rst
  assert False, 'Shape mismatch' + str(self.shape) + '+' + str(other.shape)
GPUArray.__add__ = newadd


def object_add(self, other, dst = None, shape = None, axis = 0):
  if shape is None:
    shape = self.shape
  assert axis == 0 or axis == len(shape) - 1
  tmp = self.reshape(shape)

  if dst is None:
    c = zeros_like(self)
  else:
    c = dst
  if axis == 0:
    tmp = reshape_first(tmp)
    copy_to(tmp + other, c)
  else:
    tmp = reshape_last(tmp)
    copy_to(tmp + other, c)

  return c

GPUArray.add = object_add


old_sub = GPUArray.__sub__
def newsub(self, other):
  if other.shape == self.shape:
    return old_add(self, other)
  if len(other.shape) == 2:
    rst = zeros_like(self)
    copy_to(self, rst)
    if other.shape[0] == self.shape[0] and other.shape[1] == 1:
      add_vec_to_rows(rst, other, alpha = -1)
    elif other.shape[1] == self.shape[1] and other.shape[0] == 1:
      add_vec_to_cols(rst, other, alpha = -1)
    elif self.shape[0] == other.shape[0] and self.shape[1] == 1:
      add_row_sum_to_vec(rst, other, alpha = -1)
    elif self.shape[1] == other.shape[1] and self.shape[0] == 1:
      ass_col_sum_to_vec(rst, other, alpha = -1)
    else:
      assert False, 'Shape mismatch' + str(self.shape) + '+' + str(other.shape)
    return rst
  assert False, 'Shape mismatch' + str(self.shape) + '+' + str(other.shape)
GPUArray.__sub__ = newsub

old_div = GPUArray.__div__
def newdiv(self, other):
  if np.isscalar(other):
    return old_div(self, other)
  else:
    rst = zeros_like(self)
    if other.shape[0] == self.shape[0] and other.shape[1] == 1:
      div_vec_to_rows(self, other, rst)
    elif other.shape[1] == self.shape[1] and other.shape[0] == 1:
      div_vec_to_cols(self, other, rst)
    else:
      rst = old_div(self,other)
    return rst
GPUArray.__div__ = newdiv

old_max = max
def max(input, axis = None):
  if axis is None:
    return old_max(input).astype(np.float32)
  else:
    assert axis < 2
    if axis == 0:
      rst = zeros((1, input.shape[1]), dtype=np.float32)
      col_max_reduce(rst, input)
    elif axis == 1:
      rst = zeros((input.shape[0], 1), dtype = np.float32)
      row_max_reduce(rst, input)
    return rst


def object_maxto(self, shape = None, axis = 0):
  if shape is None:
    shape = self.shape
  assert axis == 0 or axis == len(shape) -1
  tmp = self.reshape(shape)
  if axis == 0:
    tmp = reshape_first(tmp)
    c = max(tmp, axis = 1)
  else:
    tmp = reshape_last(tmp)
    c = max(tmp, axis = 0)
  return c

GPUArray.maxto = object_maxto


def argmax(input, axis):
  if axis == 0:
    rst = zeros((1, input.shape[1]), dtype = np.float32)
    find_col_max_id(rst, input)
  elif axis == 1:
    rst = zeros((input.shape[0], 1), dtype = np.float32)
    find_row_max_id(rst, input)
  else:
    assert False, 'Wrong axis'
  return rst

def object_argmaxto(self, shape = None, axis = 0):
  if shape is None:
    shape = self.shape

  assert axis == 0 or axis == len(shape) -1
  tmp = self.reshape(shape)
  if axis == 0:
    tmp = reshape_first(tmp)
    c = argmax(tmp, axis = 1)
  else:
    tmp = reshape_last(tmp)
    c = argmax(tmp, axis = 0)
  return c

GPUArray.argmaxto = object_argmaxto

def exp(input, output = None):
  if output is None:
    output = zeros_like(input)
  copy_to(input, output)
  eltwise_exp(output)
  return output

def iexp(input):
  eltwise_exp(input)

logreg_cost_col = logreg_cost_col_reduce

old_sum = gpuarray.sum
def sum(input, axis = None):
  if axis is None:
    return old_sum(input).get().item()
  else:
    assert axis < 2
    if axis == 0:
      rst = zeros((1, input.shape[1]), dtype = np.float32)
      add_col_sum_to_vec(rst, input)
    elif axis == 1:
      rst = zeros((input.shape[0], 1), dtype = np.float32)
      add_row_sum_to_vec(rst, input)
    return rst

def object_sumto(self, shape= None, axis = 0):
  if shape is None:
    shape = self.shape

  assert axis == 0 or axis == len(shape) -1
  tmp = self.reshape(shape)
  if axis == 0:
    tmp = reshape_first(tmp)
    c = sum(tmp, axis = 1)
  else:
    tmp = reshape_last(tmp)
    c = sum(tmp, axis = 0)
  return c

GPUArray.sumto = object_sumto
