import varray
from varray.ndarray import VArray, DistMethod, zeros_like, WORLD
from varray.area import Area
import garray


def copy_to(input, output):
  assert input.unique == output.unique
  assert input.shape == output.shape

  garray.copy_to(input.local_data, output.local_data)


def bigger_than_scaler(input, scaler):
  garray.bigger_than_scaler(input.local_data, scaler)

def matrixmult(x, y):
  assert isinstance(x, VArray)
  assert isinstance(y, VArray)

  if x.unique:
    x.gather()
    x.local_data = garray.reshape_last(x.local_data)
  if y.unique:
    y.gather()
    y.local_data = garray.reshape_last(y.local_data)

  c = garray.matrixmult(x.local_data, y.local_data)
  return Varray(c, unique = False)


def matrix_add(incr, grad ,alpha = 1.0, beta = 1.0):
  if len(incr.shape) == 2:
    garray.matrix_add(inrc.local_data, grad.local_data, alpha = alpha, beta = beta)
  else:
    old_shape = incr.local_data.shape
    incr.local_data = garray.reshape_last(inrc.local_data)
    garray.matrix_add(incr.local_data, garray.reshape_last(grad.local_data),
        alpha = alpha, beta = beta)
    incr.local_data = incr.local_data.reshape(old_shape)

def transpose(mat):
  assert mat.unique == False
  c = zeros_like(mat)
  c.local_data = garray.transpose(mat.local_data)
  return c

def sumto(input, shape = None, axis = 0):
  return input.sumto(shape, axis)

def maxto(input, shape = None, axis = 0):
  return input.maxto(shape, axis)


def argmaxto(input, shape = None, axis = 0):
  return input.argmaxto(shape, axis = axis)
  

def sum(input):
  return input.sum()

def exp(input):
  c = zeros_like(input)
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
  garray.relu_active(input.local_data, output.local_data, e)

def relu_compute_grad(grad, output, out_grad, e):
  garray.relu_comput_grad(grad.local_data, output.local_data, out_grad.local_data, e)

def tanh_activate(input, output, a, b):
  garray.tanh_avtivate(input.local_data, output.local_data, a, b)

def tanh_compute_grad(grad, output, out_grad, a, b):
  garray.tanh_compute_grad(grad.local_data, output.local_data, out_grad.local_data, a, b)

def convolution(input, filter ,output, image_y, output_y, output_x, padding, stride, channel, group):
  assert isinstance(input, VArray) and isinstance(filter, VArray) and isinstance(output, VArray)
  assert input.slice_method == DistMethod.Square and filter.unique == False and output.slice_method == DistMethod.Square

  input.cross_communicate(padding = padding, stride = stride, filter_size = filter.local_shape[1])

  input.pad(padding)
  r, c = output.slice_dim
  image_y = input.tmp_local_data.shape[r]
  output_y = output.local_shape[r]
  output_x = output.local_shape[c]

  old_shape = output.local_data.shape
  output.local_data = garray.reshape_last(output.local_data)

  garray.convolution(
      garray.reshape_last(input.tmp_local_data),
      garray.reshape_last(filter.local_data),
      output.local_data,
      image_y, output_y, output_x, 0, stride, channel, group)
  output.local_data = output.local_data.reshape(old_shape)


def bconvolution(input, grad, filter, out_grad, image_y, image_x, output_size, padding, stride, channel):
  assert isinstance(grad, VArray) and isinstance(filter, VArray) and isinstance(out_grad, VArray)
  assert grad.slice_method == DistMethod.Square and filter.unique == False and out_grad.slice_method == DistMethod.Square

  if not hasattr(input, 'tmp_local_data'):
    input.cross_communicate(padding = padding, stride = stride, filter_size = filter.local_shape[1])
    input.pad(padding)
  tmp_out_grad = garray.reshape_last(garray.zeros_like(input.tmp_local_data))
  r, c  = input.slice_dim
  image_y = input.tmp_local_data.shape[r]
  image_x = input.tmp_local_data.shape[c]
  output_size = grad.local_shape[r]

  garray.bconvolution(
      garray.reshape_last(input.tmp_local_data),
      garray.reshape_last(grad.local_data),
      garray.reshape_last(filter.local_data),
      tmp_out_grad,
      image_y, image_x, output_size, 0, stride, channel)

  tmp_out_grad = tmp_out_grad.reshape(input.tmp_local_data.shape)
  tmp_out_grad = input.unpad(tmp_out_grad, padding)
  out_grad.write(area = input.tmp_local_area, data = tmp_out_grad)

def wconvolution(input, grad, weight_grad, image_y, output_y, output_x, filter_size, padding, stride, channel):
  if not hasattr(input, 'tmp_local_data'):
    input.cross_communicate(padding = padding, stride = stride, filter_size = filter_size)
    input.pad(padding)
  r,c = input.slice_dim
  image_y = input.tmp_local_data.shape[r]
  output_y = grad.local_shape[r]
  output_x = grad.local_shape[c]

  tmp_weight_grad = garray.reshape_last(garray.zeros_like(weight_grad.local_data))

  garray.wconvolution(
      garray.reshape_last(input.tmp_local_data),
      garray.reshape_last(grad.local_data),
      tmp_weight_grad,
      image_y, output_y, output_x, filter_size, 0, stride, channel)

  tmp_weight_grad = tmp_weight_grad.reshape(weight_grad.local_data.shape)
  weight_grad.write(weight_grad.global_area, tmp_weight_grad)

def maxpool(input, output, channel, pool_size, start, stride, input_y, output_y, output_x):
  input.cross_communicate(stride = stride, filter_size = pool_size)
  input.pad(0)
  r,c = output.slice_dim

  output_y = output.local_shape[r]
  output_x = output.local_shape[c]
  input_y = input.tmp_local_data.shape[r]

  old_shape = output.local_data.shape
  output.local_data = garray.reshape_last(output.local_data)

  garray.maxpool(
      garray.reshape_last(input.tmp_local_data),
      output.local_data,
      channel, pool_size, start, stride, input_y, output_y, output_x)

  output.local_data = output.local_data.reshape(old_shape)

def maxundo(input, grad, output, out_grad, pool_size, start, stride, output_y, output_x, input_y):
  if not hasattr(input, 'tmp_local_data'):
    input.cross_communicate(stride = stride, filter_size = pool_size)
    input.pad(0)
  tmp_out_grad = garray.reshape_last(garray.zeros_like(input.tmp_local_data))
  r, c = input.slice_dim
  output_y = output.local_data.shape[r]
  output_x = output.local_data.shape[c]

  input_y = input.tmp_local_data.shape[r]

  garray.maxundo(
      garray.reshape_last(input.tmp_local_data),
      garray.reshape_last(grad.local_data),
      garray.reshape_last(output.local_data),
      tmp_out_grad,
      pool_size, start, stride, output_y, output_x, input_y)

  tmp_out_grad = tmp_out_grad.reshape(input.tmp_local_data.shape)
  out_grad.write(data = tmp_out_grad, area = input.tmp_local_area)

def avgpool(input, output, channel, pool_size, start, stride, input_y, output_y, output_x):
  # same as max pooling layer
  input.cross_communicate(stride = stride, filter_size = pool_size)
  input.pad(0)
  r,c = output.slice_dim

  output_y = output.local_shape[r]
  output_x = output.local_shape[c]
  input_y = input.tmp_local_data.shape[r]

  old_shape = output.local_data.shape
  output.local_data = garray.reshape_last(output.local_data)

  garray.avgpool(
      garray.reshape_last(input.tmp_local_data),
      output.local_data,
      channel, pool_size, start, stride, input_y, output_y, output_x)

  output.local_data = output.local_data.reshape(old_shape)

def avgundo(input, grad, out_grad, pool_size, start, stride, output_y, output_x, image_y, image_x):
  if not hasattr(input, 'tmp_local_data'):
    input.cross_communicate(stride = stride, filter_size = pool_size)
    input.pad(0)
  tmp_out_grad = garray.reshape_last(garray.zeros_like(input.tmp_local_data))
  r, c = input.slice_dim
  output_y = grad.local_data.shape[r]
  output_x = grad.local_data.shape[c]

  image_y = input.tmp_local_data.shape[r]
  image_x = input.tmp_local_data.shape[c]

  garray.avgundo(
      garray.reshape_last(input.tmp_local_data),
      garray.reshape_last(grad.local_data),
      tmp_out_grad,
      pool_size, start, stride, output_y, output_x, image_y, image_x)

  tmp_out_grad = tmp_out_grad.reshape(input.tmp_local_data.shape)
  out_grad.write(data = tmp_out_grad, area = input.tmp_local_area)

def rnorm(input, denom, output, channel, size, image_y, scaler, pow):
  input.cross_communicate(filter_size = size, stride = 1)
  input.pad(0)
  r, c = output.slice_dim

  tmp_out_data = garray.reshape_last(garray.zeros_like(input.tmp_local_data))
  tmp_denom_data = garray.reshape_last(garray.zeros_like(input.tmp_local_data))

  image_y = input.tmp_local_data.shape[r]

  garray.rnorm(
      garray.reshape_last(input.tmp_local_data),
      tmp_denom_data,
      tmp_out_data,
      channel, size, image_y, scaler, pow)

  denom.tmp_local_data = tmp_denom_data
  output.tmp_local_data = tmp_out_data
  output.write(input.tmp_local_area, tmp_out_data.reshape(input.tmp_local_data.shape), acc = 'no')
  denom.write(input.tmp_local_area, tmp_denom_data.reshape(input.tmp_local_data.shape), acc = 'no')

def rnormundo(grad, denom, input, output, out_grad, channel, size, image_y, scaler, pow):
  if not hasattr(input, 'tmp_local_data'):
    input.cross_communicate(stride = 1, filter_size = 1)
    input.pad(0)

    output.cross_communicate(stride = 1, filter_size = 1)
    output.pad(0)

    denom.cross_communicate(stride = 1, filter_size = 1)
    denom.pad(0)

  if not hasattr(grad, 'tmp_local_data'):
    grad.cross_communicate(stride = 1, filter_size = 1)
    grad.pad(0)

  tmp_out_grad = garray.reshape_last(garray.zeros_like(input.tmp_local_data))

  r, c = output.slice_dim
  image_y = input.tmp_local_data.shape[r]

  garray.rnormundo(
      garray.reshape_last(grad.tmp_local_data),
      garray.reshape_last(denom.tmp_local_data),
      garray.reshape_last(input.tmp_local_data),
      garray.reshape_last(output.tmp_local_data),
      tmp_out_grad,
      channel, size, image_y, scaler, pow)
  tmp_out_grad = tmp_out_grad.reshape(input.tmp_local_data.shape)
  out_grad.write(data = tmp_out_grad, area = input.tmp_local_area, acc = 'no')

def rnormcrossmap(input, denom, output, channel, size,image_y, scaler, pow, blocked):
  r, c = input.slice_dim

  denom.local_data = garray.reshape_last(denom.local_data)
  output.local_data = garray.reshape_last(output.local_data)
  image_y = input.local_data.shape[r]
  garray.rnormcrossmap(
      garray.reshape_last(input.local_data),
      denom.local_data,
      output.local_data,
      channel, size, image_y, scaler, pow, blocked)

  output.local_data = output.local_data.reshape(input.local_shape)
  denom.local_data = denom.local_data.reshape(input.local_shape)

def rnormcrossmapundo(grad, denom, input, output, out_grad, channel, size, image_y, scaler, pow, blocked):
  r, c = input.slice_dim
  image_y = input.local_data.shape[r]

  out_grad.local_data = garray.reshape_last(out_grad.local_data)

  garray.rnormcrossmapundo(
      garray.reshape_last(grad.local_data),
      garray.reshape_last(denom.local_data),
      garray.reshape_last(input.local_data),
      garray.reshape_last(output.local_data),
      out_grad.local_data,
      channel, size, image_y, scaler, pow, blocked)

  out_grad.local_data = out_grad.local_data.reshape(input.local_shape)


