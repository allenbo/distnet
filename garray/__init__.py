from pycuda import gpuarray, driver
from pycuda.gpuarray import GPUArray, to_gpu, zeros, zeros_like, empty
import numpy as np
from cuda_kernel import *
from distnet.util import divup

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



old_getitem = GPUArray.__getitem__
def new_getitem(self, index):

  if len(self.shape) > 4:
    return old_getitem(self, index)

  if not isinstance(index, tuple):
    index = (index, )

  index_axis = 0
  array_axis = 0
  slices = []
  new_shape = []
  while index_axis < len(index):
    index_entry = index[index_axis]

    if array_axis > len(self.shape):
      raise IndexError("too many axes in index")

    if isinstance(index_entry, slice):
      slices.append(index_entry)
      start, stop, idx_stride = index_entry.indices(self.shape[array_axis])
      new_shape.append(divup(stop-start, idx_stride))
    else:
      assert False

    index_axis += 1
    array_axis += 1

  while array_axis < len(self.shape):
    new_shape.append(self.shape[array_axis])
    slices.append(slice(0, self.shape[array_axis]))
    array_axis += 1

  output = GPUArray(shape = tuple(new_shape), dtype = self.dtype)
  return stride_copy(self, output, slices)

GPUArray.__getitem__ = new_getitem



def new_setitem(self, index, data):
  assert len(self.shape) <= 4, str(self.shape)

  if not isinstance(index, tuple):
    index = (index, )

  if not isinstance(data, GPUArray):
    data = array(np.require(data, dtype = self.dtype, requirements = 'C'))

  index_axis = 0
  array_axis = 0
  slices = []
  new_shape = []
  while index_axis < len(index):
    index_entry = index[index_axis]

    if array_axis > len(self.shape):
      raise IndexError("too many axes in index")

    if isinstance(index_entry, slice):
      slices.append(index_entry)
      start, stop, idx_stride = index_entry.indices(self.shape[array_axis])
      new_shape.append(divup(stop-start, idx_stride))
    else:
      assert False

    index_axis += 1
    array_axis += 1

  while array_axis < len(self.shape):
    new_shape.append(self.shape[array_axis])
    slices.append(slice(0, self.shape[array_axis]))
    array_axis += 1

  assert data.shape == tuple(new_shape)

  stride_write(data, self, slices)

GPUArray.__setitem__ = new_setitem


def concatenate(arrays, axis = 0):
  if not isinstance(arrays, tuple):
    raise TypeError('First parameter has to be a tuple of arrays')

  other_dim = arrays[0].shape[:axis] + arrays[0].shape[axis+1:]
  for array in arrays:
    if axis >= len(array.shape):
      raise RuntimeError('axis is too big, axis = %d, shape = %s' % ( axis, array.shape))
    if array.shape[:axis] + array.shape[axis+1:] != other_dim:
      raise ValueError('array dimensions must agree expect for d_%d' % axis)
  source = arrays[0]
  for other in arrays[1:]:
    new_shape = source.shape[:axis] + (source.shape[axis] + other.shape[axis], ) + source.shape[axis+1:]

    dest = GPUArray(shape = tuple(new_shape), dtype = source.dtype)

    slices = tuple([slice(0, j, 1) for j in source.shape])
    dest[slices] = source

    extend = source.shape[axis]
    slices = [slice(0, j, 1) for j in other.shape]
    slices[axis] = slice(extend, extend+ other.shape[axis], 1)

    slices = tuple(slices)
    dest[slices] = other

    source = dest
  return dest

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
