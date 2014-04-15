from distbase import util
from distbase.util import timer, divup, make_copy
from distbase.cuda_base import matrixmult, transpose
from pycuda import driver
from pycuda.gpuarray import GPUArray
from distbase import cuda_base
from distbase.util import divup, make_copy
from distbase.cuda_base import *

def sync_function(fn):
  def _wrapper(*args, **kw):
    result = fn(*args, **kw)
    driver.Context.synchronize()
    return result
  return make_copy(_wrapper, fn.__name__)

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

def convert_shape_fn(fn):

  def convert_shape(arg):
    if isinstance(arg, GPUArray) and len(arg.shape) > 2:
      shape_len = len(arg.shape)
      fd = int(np.prod(arg.shape[:shape_len/2]))
      sd = int(np.prod(arg.shape[shape_len/2:]))
      arg = arg.reshape((fd, sd))
    return arg

  def _fn(*args, **kw):
    cargs = [convert_shape(arg) for arg in args]
    ckw = {key:convert_shape(value) for key, value in kw.iteritems()}
    fn(*cargs, **ckw)

  return make_copy(_fn, fn.__name__)

same_reduce = convert_shape_fn(cuda_base.same_reduce)
same_reduce_multiview = convert_shape_fn(cuda_base.same_reduce_multiview)
relu_activate = convert_shape_fn(cuda_base.relu_activate)
relu_compute_grad = convert_shape_fn(cuda_base.relu_compute_grad)
tanh_activate = convert_shape_fn(cuda_base.tanh_activate)
tanh_compute_grad = convert_shape_fn(cuda_base.tanh_compute_grad)
gpu_copy_to = convert_shape_fn(cuda_base.gpu_copy_to)
gpu_partial_copy_to = convert_shape_fn(cuda_base.gpu_partial_copy_to)
matrix_add = convert_shape_fn(cuda_base.matrix_add)
bigger_than_scalar = convert_shape_fn(cuda_base.bigger_than_scalar)
eltwise_exp = convert_shape_fn(cuda_base.eltwise_exp)
eltwise_mul = convert_shape_fn(cuda_base.eltwise_mul)
