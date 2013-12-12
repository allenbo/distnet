import varray
import time
import util
import numpy as np
from varray.ndarray import VArray, DistMethod, zeros_like, WORLD, zeros, allocate_like, allocate,\
WORLD
from varray.area import Area
import garray
from pycuda import driver


gpu_cache = {}

def get_from_cache(shape):
  if shape in gpu_cache:
    c = gpu_cache[shape]
    del gpu_cache[shape]
  else:
    c = garray.GPUArray(shape, dtype = np.float32)

  return c

def put_to_cache(shape, c):
  gpu_cache[shape] = c

def copy_to(input, output):
  if output.unique:
    if not input.unique:
      output.copy_from_global(input.local_data.reshape(output.shape))
      return

  garray.copy_to(input.local_data, output.local_data)


def partial_copy(input, f, t):
  ''' partial copy last dimention '''
  shape = list(input.shape)
  shape[-1] = t-f
  rst = allocate(tuple(shape), dtype = np.float32)
  old_shape = rst.local_shape
  rst.local_data = garray.partial_copy(garray.reshape_last(input.local_data), f, t)
  rst.local_data = rst.local_data.reshape(old_shape)
  return rst


def bigger_than_scaler(input, scaler):
  garray.bigger_than_scaler(input.local_data, scaler)

def matrixmult(x, y, dest = None):
  assert isinstance(x, VArray)
  assert isinstance(y, VArray)

  if x.unique:
    x = garray.reshape_last(x.fetch(x.global_area))
  else:
    x = x.local_data
  if y.unique:
    y = garray.reshape_last(y.fetch(y.global_area))
  else:
    y = y.local_data
  shape = (x.shape[0], y.shape[1])
  #c = garray.matrixmult(x, y, atrans = atrans, btrans = btrans)
  if dest is None or dest.unique == True:
    c = get_from_cache(shape)
    garray.matrixmult(x, y, c)
    rst = VArray(c, unique = False)
    put_to_cache(shape, c)
    return rst
  else:
    garray.matrixmult(x, y, dest.local_data)
    return dest


def matrix_add(incr, grad ,alpha = 1.0, beta = 1.0):
  if len(incr.shape) == 2:
    garray.matrix_add(incr.local_data, grad.local_data, alpha = alpha, beta = beta)
  else:
    old_shape = incr.local_data.shape
    incr.local_data = garray.reshape_last(incr.local_data)
    garray.matrix_add(incr.local_data, garray.reshape_last(grad.local_data),
        alpha = alpha, beta = beta)
    incr.local_data = incr.local_data.reshape(old_shape)

def transpose(mat):
  if mat.unique:
    x = garray.reshape_last(mat.fetch(mat.global_area))
  else:
    x = mat.local_data
  c = garray.transpose(x)
  return VArray(c, unique = False)

def sumto(input, shape = None, axis = 0):
  return input.sumto(shape, axis)

def maxto(input, shape = None, axis = 0):
  return input.maxto(shape, axis)


def argmaxto(input, shape = None, axis = 0):
  return input.argmaxto(shape, axis = axis)
  

def sum(input):
  return input.sum()

def exp(input):
  c = allocate_like(input)
  garray.copy_to(input.local_data, c.local_data)
  return c

def iexp(input):
  garray.iexp(input.local_data)

def logreg_cost_col(output, label, cost):
  assert not any([output.unique, label.unique, cost.unique])
  garray.logreg_cost_col(output.local_data, label.local_data, cost.local_data)

def max(input):
  return input.max()

def softmax_bprop(output, label, out_grad):
  garray.softmax_bprop(output.local_data, label.local_data, out_grad.local_data)

def relu_activate(input, output, e):
  if len(input.local_shape) != 2:
    old_shape = output.local_shape
    output.local_data = garray.reshape_last(output.local_data)
    garray.relu_activate(
        garray.reshape_last(input.local_data),
        output.local_data,
        e)
    output.local_data = output.local_data.reshape(old_shape)
  else:
    garray.relu_activate(
        input.local_data,
        output.local_data,
        e)


def relu_compute_grad(grad, output, out_grad, e):
  if len(grad.shape) != 2:
    old_shape = out_grad.local_shape
    out_grad.local_data = garray.reshape_last(out_grad.local_data)
    garray.relu_compute_grad(
        garray.reshape_last(grad.local_data),
        garray.reshape_last(output.local_data),
        out_grad.local_data,
        e)

    out_grad.local_data = out_grad.local_data.reshape(old_shape)
  else:
    garray.relu_compute_grad(grad.local_data, output.local_data, out_grad.local_data, e)

def tanh_activate(input, output, a, b):
  garray.tanh_avtivate(input.local_data, output.local_data, a, b)

def tanh_compute_grad(grad, output, out_grad, a, b):
  garray.tanh_compute_grad(grad.local_data, output.local_data, out_grad.local_data, a, b)

def convolution(input, filter ,output, image_y, output_y, output_x, padding, stride, channel, group):
  assert isinstance(input, VArray) and isinstance(filter, VArray) and isinstance(output, VArray)
  assert input.slice_method == DistMethod.Square and filter.unique == False and output.slice_method == DistMethod.Square

  input.cross_communicate(padding = padding, stride = stride, filter_size = filter.local_shape[1])

  r, c = output.slice_dim
  image_y = input.tmp_local_data.shape[r]
  output_y = output.local_shape[r]
  output_x = output.local_shape[c]

  garray.convolution(
      input.tmp_local_data,
      filter.local_data,
      output.local_data,
      image_y, output_y, output_x, 0, stride, channel, group)



def bconvolution(input, grad, filter, out_grad, image_y, image_x, output_size, padding, stride, channel):
  assert isinstance(grad, VArray) and isinstance(filter, VArray) and isinstance(out_grad, VArray)
  assert grad.slice_method == DistMethod.Square and filter.unique == False and out_grad.slice_method == DistMethod.Square

  start = time.time()
  if not hasattr(input, 'tmp_local_data'):
    input.cross_communicate(padding = padding, stride = stride, filter_size = filter.local_shape[1])
  if not hasattr(out_grad, 'tmp_out_grad'):
    tmp_out_grad = garray.empty_like(input.tmp_local_data)
    out_grad.tmp_out_grad = tmp_out_grad
  else:
    tmp_out_grad = out_grad.tmp_out_grad
  r, c  = input.slice_dim
  image_y = input.tmp_local_data.shape[r]
  image_x = input.tmp_local_data.shape[c]
  output_size = grad.local_shape[r]

  garray.bconvolution(
      input.tmp_local_data,
      grad.local_data,
      filter.local_data,
      tmp_out_grad,
      image_y, image_x, output_size, 0, stride, channel)

  tmp_out_grad = input.unpad(tmp_out_grad, padding)
  out_grad.write(area = input.tmp_local_area, data = tmp_out_grad)

def wconvolution(input, grad, weight_grad, image_y, output_y, output_x, filter_size, padding, stride, channel):
  if not hasattr(input, 'tmp_local_data'):
    input.cross_communicate(padding = padding, stride = stride, filter_size = filter_size)
  r,c = input.slice_dim
  image_y = input.tmp_local_data.shape[r]
  output_y = grad.local_shape[r]
  output_x = grad.local_shape[c]

  if not hasattr(weight_grad, 'tmp_out_grad'):
    tmp_weight_grad = garray.GPUArray(weight_grad.shape, dtype = weight_grad.dtype)
    weight_grad.tmp_weight_grad = tmp_weight_grad
  else:
    tmp_weight_grad = out_weight.tmp_weight_grad

  garray.wconvolution(
      input.tmp_local_data,
      grad.local_data,
      tmp_weight_grad,
      image_y, output_y, output_x, filter_size, 0, stride, channel)

  weight_grad.write(weight_grad.global_area, tmp_weight_grad)

def maxpool(input, output, channel, pool_size, start, stride, input_y, output_y, output_x):
  r,c = output.slice_dim
  num_row = output.local_shape[r]
  num_col = output.local_shape[c]
  input.cross_communicate(stride = stride, filter_size = pool_size, num_output = (num_row, num_col))

  output_y = output.local_shape[r]
  output_x = output.local_shape[c]
  input_y = input.tmp_local_data.shape[r]

  old_shape = output.local_data.shape

  garray.maxpool(
      input.tmp_local_data,
      output.local_data,
      channel, pool_size, start, stride, input_y, output_y, output_x)


def maxundo(input, grad, output, out_grad, pool_size, start, stride, output_y, output_x, input_y):
  r, c = input.slice_dim
  if not hasattr(input, 'tmp_local_data'):
    num_row = output.local_shape[r]
    num_col = output.local_shape[c]
    input.cross_communicate(stride = stride, filter_size = pool_size, num_output = (num_row, num_col))
  if not hasattr(out_grad, 'tmp_out_grad'):
    tmp_out_grad = garray.empty_like(input.tmp_local_data)
    out_grad.tmp_out_grad = tmp_out_grad
  else:
    tmp_out_grad = out_grad.tmp_out_grad
  output_y = output.local_data.shape[r]
  output_x = output.local_data.shape[c]

  input_y = input.tmp_local_data.shape[r]

  garray.maxundo(
      input.tmp_local_data,
      grad.local_data,
      output.local_data,
      tmp_out_grad,
      pool_size, start, stride, output_y, output_x, input_y)

  out_grad.write(data = tmp_out_grad, area = input.tmp_local_area)

def avgpool(input, output, channel, pool_size, start, stride, input_y, output_y, output_x):
  # same as max pooling layer
  r,c = output.slice_dim
  num_row = output.local_shape[r]
  num_col = output.local_shape[c]
  input.cross_communicate(stride = stride, filter_size = pool_size, num_output = (num_row, num_col))

  output_y = output.local_shape[r]
  output_x = output.local_shape[c]
  input_y = input.tmp_local_data.shape[r]

  old_shape = output.local_data.shape

  garray.avgpool(
      input.tmp_local_data,
      output.local_data,
      channel, pool_size, start, stride, input_y, output_y, output_x)

def avgundo(input, grad, out_grad, pool_size, start, stride, output_y, output_x, image_y, image_x):
  r, c = input.slice_dim
  if not hasattr(input, 'tmp_local_data'):
    num_row = output.local_shape[r]
    num_col = output.local_shape[c]
    input.cross_communicate(stride = stride, filter_size = pool_size, num_output = (num_row, num_col))
  if not hasattr(out_grad, 'tmp_out_grad'):
    tmp_out_grad = garray.empty_like(input.tmp_local_data)
    out_grad.tmp_out_grad = tmp_out_grad
  else:
    tmp_out_grad = out_grad.tmp_out_grad
  tmp_out_grad = garray.empty_like(input.tmp_local_data)
  output_y = grad.local_data.shape[r]
  output_x = grad.local_data.shape[c]

  image_y = input.tmp_local_data.shape[r]
  image_x = input.tmp_local_data.shape[c]

  garray.avgundo(
      input.tmp_local_data,
      grad.local_data,
      tmp_out_grad,
      pool_size, start, stride, output_y, output_x, image_y, image_x)

  out_grad.write(data = tmp_out_grad, area = input.tmp_local_area)

def rnorm(input, denom, output, channel, size, image_y, scaler, pow):
  input.cross_communicate(filter_size = size, stride = 1)
  r, c = output.slice_dim

  tmp_out_data = garray.empty_like(input.tmp_local_data)
  tmp_denom_data = garray.empty_like(input.tmp_local_data)

  image_y = input.tmp_local_data.shape[r]

  garray.rnorm(
      input.tmp_local_data,
      tmp_denom_data,
      tmp_out_data,
      channel, size, image_y, scaler, pow)

  denom.tmp_local_data = tmp_denom_data
  output.tmp_local_data = tmp_out_data
  output.write(input.tmp_local_area, tmp_out_data, acc = 'no')
  denom.write(input.tmp_local_area, tmp_denom_data, acc = 'no')

def rnormundo(grad, denom, input, output, out_grad, channel, size, image_y, scaler, pow):
  if not hasattr(input, 'tmp_local_data'):
    input.cross_communicate(stride = 1, filter_size = size)
    denom.cross_communicate(stride = 1, filter_size = size)

  if output.tmp_local_data.shape != input.tmp_local_data.shape:
    output.cross_communicate(stride = 1, filter_size = size)


  if not hasattr(grad, 'tmp_local_data'):
    grad.cross_communicate(stride = 1, filter_size = size)

  tmp_out_grad = garray.empty_like(input.tmp_local_data)

  r, c = output.slice_dim
  image_y = input.tmp_local_data.shape[r]

  garray.rnormundo(
      grad.tmp_local_data,
      denom.tmp_local_data,
      input.tmp_local_data,
      output.tmp_local_data,
      tmp_out_grad,
      channel, size, image_y, scaler, pow)
  out_grad.write(data = tmp_out_grad, area = input.tmp_local_area, acc = 'no')

def rnormcrossmap(input, denom, output, channel, size,image_y, scaler, pow, blocked):
  r, c = input.slice_dim

  image_y = input.local_data.shape[r]
  garray.rnormcrossmap(
      input.local_data,
      denom.local_data,
      output.local_data,
      channel, size, image_y, scaler, pow, blocked)

def rnormcrossmapundo(grad, denom, input, output, out_grad, channel, size, image_y, scaler, pow, blocked):
  r, c = input.slice_dim
  image_y = input.local_data.shape[r]

  garray.rnormcrossmapundo(
      grad.local_data,
      denom.local_data,
      input.local_data,
      output.local_data,
      out_grad.local_data,
      channel, size, image_y, scaler, pow, blocked)
