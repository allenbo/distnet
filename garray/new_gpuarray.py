from pycuda import gpuarray, driver
from pycuda.gpuarray import GPUArray, to_gpu, zeros, zeros_like, empty, empty_like, to_gpu_async
import numpy as np
from aux_operation import *
from .backend import backend_name
from distbase.util import divup, make_copy, deprecated, random_string
from distbase import cuda_base
import time
import traceback

default_stream = None

@sync_function
def array(obj, dtype = None, to2dim = False):
  global default_stream
  if default_stream is None:
      default_stream = driver.Stream()
  if dtype is None:
    dtype = obj.dtype
  if not isinstance(obj, GPUArray):
    obj = to_gpu(obj).astype(dtype)
  if len(obj.shape) != 2 and len(obj.shape) != 1 and to2dim:
    obj = reshape_last(obj)
  obj.stream = default_stream
  return obj

copy_to = sync_function(gpu_copy_to)
partial_copy_to = sync_function(gpu_partial_copy_to)
logreg_cost_col = logreg_cost_col_reduce


def get_mem_info():
  return tuple([x / 1024.0  for x in driver.mem_get_info()])

old_init = GPUArray.__init__
def new_init(self, *args, **kw):
  global default_stream
  if default_stream is None:
      default_stream = driver.Stream()
  #traceback.print_stack()
  result = old_init(self, *args, **kw)
  driver.Context.synchronize()
  self.stream = default_stream
  return result
GPUArray.__init__ = new_init


old_getitem = GPUArray.__getitem__
@sync_function
def new_getitem(self, index):
  if len(self.shape) > 4:
    return old_getitem(self, index)

  if not isinstance(index, tuple):
    index = (index, )
  start = time.time()
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
  result = stride_copy(self, output, slices)
  return result
GPUArray.__getitem__ = new_getitem


@sync_function
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

old_add = GPUArray.__add__
@sync_function
def newadd(self, other):
  if other.shape == self.shape:
    return old_add(self, other)
  # Only allow other's shape to be 2D. Other operations don't deliver any value to distnet
  if len(other.shape) == 2:
    rst = empty_like(self)
    copy_to(self, rst)
    # shape will match when other.shape[0] == self.shape[0] or other.shape[1] == self.shape[-1]
    if other.shape[0] == self.shape[0] and other.shape[1] == 1:
      add_vec_to_rows(reshape_first(rst), other)
    elif other.shape[1] == self.shape[-1] and other.shape[0] == 1:
      add_vec_to_cols(reshape_last(rst), other)
    else:
      assert False, 'Shape mismatch' + str(self.shape) + '+' + str(other.shape)

    return rst
  assert False, 'Shape mismatch' + str(self.shape) + '+' + str(other.shape)
GPUArray.__add__ = newadd

old_sub = GPUArray.__sub__
@sync_function
def newsub(self, other):
  # like newadd, remove trivial code
  if other.shape == self.shape:
    return old_sub(self, other)
  if len(other.shape) == 2:
    rst = empty_like(self)
    copy_to(self, rst)
    if other.shape[0] == self.shape[0] and other.shape[1] == 1:
      add_vec_to_rows(reshape_first(rst), other, alpha = -1)
    elif other.shape[1] == self.shape[-1] and other.shape[0] == 1:
      add_vec_to_cols(reshape_last(rst), other, alpha = -1)
    else:
      assert False, 'Shape mismatch' + str(self.shape) + '+' + str(other.shape)
    return rst
  assert False, 'Shape mismatch' + str(self.shape) + '+' + str(other.shape)
GPUArray.__sub__ = newsub


old_div = GPUArray.__div__
@sync_function
def newdiv(self, other):
  if np.isscalar(other):
    return old_div(self, other)
  else:
    rst = empty_like(self)
    if other.shape[0] == self.shape[0] and other.shape[1] == 1:
      div_vec_to_rows(reshape_first(self), other, rst)
    elif other.shape[1] == self.shape[-1] and other.shape[0] == 1:
      div_vec_to_cols(reshape_last(self), other, rst)
    else:
      rst = old_div(self,other)
    return rst
GPUArray.__div__ = newdiv

@sync_function
def setitem_sum(self, index, data):
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
  stride_write_sum(data, self, slices)
GPUArray.setitem_sum = setitem_sum

old_sum = gpuarray.sum
@sync_function
def sum(input, axis = None):
  '''
  This function only accommodate with 2D array
  TODO: support 4D operation
  '''
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

@deprecated
@sync_function
def object_sumto(self, shape= None, axis = 0):
  if shape is None:
    shape = self.shape
  assert axis >= 0 and axis < len(shape)

  tmp = self.reshape(shape) if self.shape != shape else self

  if axis == 0:
    tmp = reshape_first(tmp) if len(tmp.shape) != 2 else tmp
    c = sum(tmp, axis = 1)
  elif axis == len(shape) -1:
    tmp = reshape_last(tmp) if len(tmp.shape) != 2 else tmp
    c = sum(tmp, axis = 0)
  else:
    sd = int(np.prod(tmp.shape[axis+1:]))
    c = gpuarray.zeros((tmp.shape[axis], 1), dtype = self.dtype)
    for i in range(np.prod(tmp.shape[:axis])):
      partial = gpuarray.GPUArray(shape = (tmp.shape[axis], sd), dtype = tmp.dtype, gpudata = tmp.ptr + i * tmp.strides[axis])
      partial_rst = sum(partial, 1)
      c += partial_rst
  return c
GPUArray.sumto = object_sumto

@deprecated
@sync_function
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
  dest = source
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

@sync_function
def partial_copy1(input, f, t):
  '''
  This function only supports 2D array and copy partial array by splitting the second dimension
  '''
  shape = list(input.shape)
  shape[-1] = t - f
  data = empty(tuple(shape), dtype = input.dtype)
  gpu_partial_copy_to(input, data, 0, shape[0], f, t)
  return data

@sync_function
def partial_copy0(input, f, t):
  '''
  This function only supports 2D array and copy partial array by splitting the first dimension
  '''
  shape = list(input.shape)
  shape[0] = t - f
  data = empty(tuple(shape), dtype = input.dtype)
  gpu_partial_copy_to(input, data, f, t, 0, shape[-1])
  return data

@deprecated
@sync_function
def object_add(self, other, dst = None, shape = None, axis = 0):
  if shape is None:
    shape = self.shape
  assert len(shape) >= len(other.shape), (shape, other, shape)
  assert axis >= 0 and axis < len(shape)

  tmp = self.reshape(shape) if self.shape != shape else self

  if dst is None:
    c = empty_like(self)
  else:
    c = dst
  tmp_shape = tmp.shape
  assert len(tmp_shape) >= 2

  if len(tmp_shape) == 2:
    copy_to(tmp + other, c)
  else:
    if axis == 0:
      tmp = reshape_first(tmp)
      copy_to(tmp + other, c)
    elif axis == len(tmp_shape) - 1:
      tmp = reshape_last(tmp)
      copy_to(tmp + other, c)
    else:
      #sd = int(np.prod(tmp_shape[axis+1:]))
      #for i in range(np.prod(tmp_shape[:axis])):
      #  partial = gpuarray.GPUArray(shape = (tmp_shape[axis], sd), dtype = tmp.dtype, gpudata = tmp.ptr + i * tmp.strides[axis])
      #  partial_dest = gpuarray.GPUArray(shape = (tmp_shape[axis], sd), dtype = tmp.dtype, gpudata = c.ptr + i * tmp.strides[axis])
      #  copy_to(partial + other, partial_dest)
      fd = int(np.prod(tmp_shape[:axis]))
      sd = int(np.prod(tmp_shape[axis:]))
      intermidiate_shape = tuple(tmp_shape[axis:] + tmp_shape[:axis])
      transpose_tmp = transpose(tmp.reshape((fd, sd))).reshape(intermidiate_shape)
      transpose_tmp = reshape_first(transpose_tmp)
      rst_tmp = transpose_tmp + other
      rst_tmp = reshape_last(rst_tmp.reshape(intermidiate_shape))
      copy_to(transpose(rst_tmp), c)
  if dst is None:
    c = c.reshape(self.shape)
  return c
GPUArray.add = object_add

old_max = max
@sync_function
def max(input, axis = None):
  '''
  This function supports only 2D array
  '''
  assert axis < 2
  if axis is None:
    return old_max(input).astype(np.float32)
  else:
    assert len(input.shape) <= 2
    if axis == 0:
      rst = empty((1, input.shape[1]), dtype=np.float32)
      col_max_reduce(rst, input)
    elif axis == 1:
      rst = empty((input.shape[0], 1), dtype = np.float32)
      row_max_reduce(rst, input)
    return rst

@deprecated
@sync_function
def object_maxto(self, shape = None, axis = 0):
  if shape is None:
    shape = self.shape
  assert axis == 0 or axis == len(shape) -1
  tmp = self.reshape(shape) if self.shape != shape else self
  if axis == 0:
    tmp = reshape_first(tmp) if len(tmp.shape)!= 2 else tmp
    c = max(tmp, axis = 1)
  else:
    tmp = reshape_last(tmp) if len(tmp.shape)!=2 else tmp
    c = max(tmp, axis = 0)
  return c
GPUArray.maxto = object_maxto


@sync_function
def argmax(input, axis):
  '''
  This function only supports 2D array
  '''
  assert len(input.shape) == 2
  if axis == 0:
    rst = empty((1, input.shape[1]), dtype = np.float32)
    find_col_max_id(rst, input)
  elif axis == 1:
    rst = empty((input.shape[0], 1), dtype = np.float32)
    find_row_max_id(rst, input)
  else:
    assert False, 'Wrong axis'
  return rst

@deprecated
@sync_function
def object_argmaxto(self, shape = None, axis = 0):
  if shape is None:
    shape = self.shape

  assert axis == 0 or axis == len(shape) -1
  tmp = self.reshape(shape) if self.shape != shape else self
  if axis == 0:
    tmp = reshape_first(tmp) if len(tmp.shape) != 2 else tmp
    c = argmax(tmp, axis = 1)
  else:
    tmp = reshape_last(tmp) if len(tmp.shape) != 2 else tmp
    c = argmax(tmp, axis = 0)
  return c
GPUArray.argmaxto = object_argmaxto

@sync_function
def exp(input, output = None):
  if output is None:
    output = empty_like(input)
  copy_to(input, output)
  eltwise_exp(output)
  return output

@sync_function
def iexp(input):
  eltwise_exp(input)

def mem_free(self):
  return 0
  self.gpudata.free()
GPUArray.mem_free = mem_free

def printout(self, name, row_from = 0, row_to = 0, col_from = 0, col_to = 0, fc = False):
  print name
  if backend_name == 'caffe' and fc == False:
    x = reshape_first(self)
    x = cuda_base.transpose(x)
  else:
    x = reshape_last(self)
  if row_to == 0:
    row_to = x.shape[0]
  if col_to == 0:
    col_to = 10#x.shape[1]
  a = x.get()[row_from: row_to , col_from: col_to]

  for rows in a:
    for i in rows:
      print '%.8f' % i,
    print ''
GPUArray.printout = printout

def dump(self):
  x = self.get()
  filename = 'arrayout-' + random_string(6)
  import cPickle as pickle
  with open(filename) as f:
    pickle.dump(x, f)
