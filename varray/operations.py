import varray
import time
from distbase import util
import numpy as np
from varray.ndarray import VArray, DistMethod, zeros_like, WORLD, zeros, allocate_like, allocate, WORLD
from varray.area import Area
import garray
from pycuda import driver
from distbase.state import *
from garray import ConvDataLayout, FCDataLayout, FilterLayout, WeightLayout

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
    x = x.fetch(x.global_area)
  else:
    x = x.local_data
  if y.unique:
    y = y.fetch(y.global_area)
  else:
    y = y.local_data
  shape = (x.shape[0], y.shape[1])
  #c = garray.matrixmult(x, y, atrans = atrans, btrans = btrans)
  if dest is None or dest.unique == True:
    c = garray.matrixmult(x, y)
    rst = VArray(c, unique = False)
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

  filter_size_index = FilterLayout.HEIGHT 
  r, c = ConvDataLayout.HEIGHT, ConvDataLayout.WIDTH
  state = get_state_from_distribution(output.slice_dim, conv = True)

  if state == disw_i:
    assert filter.unique == False
    input.image_communicate(slice_dim = (r, c), padding = padding, stride = stride, filter_size = filter.local_shape[filter_size_index], output_area = output.local_area)
  elif state == disw_b:
    input.batch_communicate(input.rank, ConvDataLayout.BATCH)
  elif state == sidw or state == sisw:
    input.global_communicate()

  input_data = input.tmp_local_data
  image_y = input_data.shape[r]
  output_y = output.local_shape[r]
  output_x = output.local_shape[c]

  garray.convolution(
      input_data,
      filter.local_data,
      output.local_data,
      image_y, output_y, output_x, 0, stride, channel, group)

def bconvolution(input, grad, filter, out_grad, image_y, image_x, output_size, padding, stride, channel):
  assert isinstance(grad, VArray) and isinstance(filter, VArray) and isinstance(out_grad, VArray)

  propagate = True
  filter_size_index = FilterLayout.HEIGHT 
  r, c = ConvDataLayout.HEIGHT, ConvDataLayout.WIDTH
  state = get_state_from_distribution(grad.slice_dim, conv = True)

  if state == disw_i:
    assert filter.unique == False
    if not hasattr(input, 'tmp_local_data'):
      input.image_communicate(slice_dim = (r, c), padding = padding, stride = stride, filter_size = filter.local_shape[filter_size_index], output_area = output.local_area)
  elif state == disw_b:
    if not hasattr(input, 'tmp_local_data'):
      input.batch_communicate(input.rank, ConvDataLayout.BATCH)
  elif state == sidw or state == sisw:
    if not hasattr(input, 'tmp_local_data'):
      input.global_communicate()
    if state == sisw:
      propagate = False
    
  input_data = input.tmp_local_data

  if not hasattr(out_grad, 'tmp_local_data'):
    out_grad.tmp_local_data = garray.empty_like(input_data)
  tmp_out_grad = out_grad.tmp_local_data

  image_y = input_data.shape[r]
  image_x = input_data.shape[c]
  output_size = grad.local_shape[r]

  garray.bconvolution(
      input_data,
      grad.local_data,
      filter.local_data,
      tmp_out_grad,
      image_y, image_x, output_size, 0, stride, channel)

  tmp_out_grad = input.unpad(tmp_out_grad, padding)
  out_grad.write(area = input.tmp_local_area, data = tmp_out_grad, propagate = propagate)

def wconvolution(input, grad, weight_grad, image_y, output_y, output_x, filter_size, padding, stride, channel):

  propagate = True
  filter_size_index = FilterLayout.HEIGHT
  r, c = ConvDataLayout.HEIGHT, ConvDataLayout.WIDTH
  state = get_state_from_distribution(grad.slice_dim, conv = True)

  if state == disw_i:
    assert filter.unique == False
    if not hasattr(input, 'tmp_local_data'):
      input.image_communicate(slice_dim = (r, c), padding = padding, stride = stride, filter_size = filter.local_shape[filter_size_index], output_area = output.local_area)
  elif state == disw_b:
    if not hasattr(input, 'tmp_local_data'):
      input.batch_communicate(input.rank, ConvDataLayout.BATCH)
  elif state == sidw or state == sisw:
    if not hasattr(input, 'tmp_local_data'):
      input.global_communicate()
  
  input_data = input.tmp_local_data

  if state in [disw_i, disw_b]:
    if not hasattr(weight_grad, 'tmp_local_data'):
      weight_grad.tmp_local_data = garray.GPUArray(weight_grad.shape, dtype = weight_grad.dtype)
    tmp_weight_grad = weight_grad.tmp_local_data
  else:
    propagate = False
    tmp_weight_grad = weight_grad.local_data

  image_y = input_data.shape[r]
  output_y = grad.local_shape[r]
  output_x = grad.local_shape[c]

  garray.wconvolution(
      input_data,
      grad.local_data,
      tmp_weight_grad,
      image_y, output_y, output_x, filter_size, 0, stride, channel)

  weight_grad.write(area = weight_grad.global_area, data = tmp_weight_grad, propagate = propagate)

def maxpool(input, output, channel, pool_size, start, stride, input_y, output_y, output_x):
  r, c = ConvDataLayout.HEIGHT, ConvDataLayout.WIDTH
  state = get_state_from_distribution(output.slice_dim, conv = True)

  if state == disw_i:
    input.image_communicate(slice_dim = (r, c), stride = stride, filter_size = pool_size, output_area = output.local_area)
  elif state == disw_b:
    input.batch_communicate(input.rank, ConvDataLayout.BATCH)
  elif state == sisw:
    input.global_communicate()
  elif state == sidw:
    input.channel_communicate(input.rank, ConvDataLayout.CHANNEL)

  input_data = input.tmp_local_data

  output_y = output.local_shape[r]
  output_x = output.local_shape[c]
  input_y = input_data.shape[r]

  garray.maxpool(
      input_data,
      output.local_data,
      channel, pool_size, start, stride, input_y, output_y, output_x)

def maxundo(input, grad, output, out_grad, pool_size, start, stride, output_y, output_x, input_y):
  propagate = True
  r, c = ConvDataLayout.HEIGHT, ConvDataLayout.WIDTH
  state = get_state_from_distribution(output.slice_dim, conv = True)

  if state == disw_i:
    if not hasattr(input, 'tmp_local_data'):
      input.image_communicate(slice_dim = (r, c), stride = stride, filter_size = pool_size, output_area = grad.local_area)
  elif state == disw_b:
    if not hasattr(input, 'tmp_local_data'):
      input.batch_communicate(input.rank, ConvDataLayout.BATCH)
  elif state == sidw:
    if not hasattr(input, 'tmp_local_data'):
      input.channel_communicate(input.rank, ConvDataLayout.CHANNEL)
  elif  state == sisw:
    if not hasattr(input, 'tmp_local_data'):
      input.global_communicate()
    propagate = False
  
  input_data = input.tmp_local_data

  if not hasattr(out_grad, 'tmp_local_data'):
    out_grad.tmp_local_data = garray.empty_like(input_data)
  tmp_out_grad = out_grad.tmp_local_data

  output_y = output.local_data.shape[r]
  output_x = output.local_data.shape[c]

  input_y = input_data.shape[r]

  garray.maxundo(
      input_data,
      grad.local_data,
      output.local_data,
      tmp_out_grad,
      pool_size, start, stride, output_y, output_x, input_y)

  out_grad.write(data = tmp_out_grad, area = input.tmp_local_area, propagate = propagte)

def avgpool(input, output, channel, pool_size, start, stride, input_y, output_y, output_x):
  r, c = ConvDataLayout.HEIGHT, ConvDataLayout.WIDTH
  state = get_state_from_distribution(output.slice_dim, conv = True)
  
  if state == disw_i:
    assert filter.unique == False
    input.image_communicate(slice_dim = (r, c), stride = stride, filter_size = pool_size, output_area = output.local_area)
  elif state == disw_b:
    input.batch_communicate(input.rank, ConvDataLayout.BATCH)
  elif state == sisw:
    input.global_communicate()
  elif state == sidw:
    input.channel_communicate(input.rank, ConvDataLayout.CHANNEL)

  input_data = input.tmp_local_data

  output_y = output.local_shape[r]
  output_x = output.local_shape[c]
  input_y = input_data.shape[r]

  garray.avgpool(
      input_data,
      output.local_data,
      channel, pool_size, start, stride, input_y, output_y, output_x)

def avgundo(input, grad, out_grad, pool_size, start, stride, output_y, output_x, image_y, image_x):
  propagate = True
  r, c = ConvDataLayout.HEIGHT, ConvDataLayout.WIDTH
  state = get_state_from_distribution(output.slice_dim, conv = True)

  if state == disw_i:
    if not hasattr(input, 'tmp_local_data'):
      input.image_communicate(slice_dim = (r, c), stride = stride, filter_size = pool_size, output_area = grad.local_area)
  elif state == disw_b:
    if not hasattr(input, 'tmp_local_data'):
      input.batch_communicate(input.rank, ConvDataLayout.BATCH)
  elif state == sidw:
    if not hasattr(input, 'tmp_local_data'):
      input.channle_communicate(input.rank, ConvDataLayout.CHANNEL)
  elif state == sisw:
    if not hasattr(input, 'tmp_local_data'):
      input.global_communicate()
    propagate = False
  
  input_data = input.tmp_local_data

  if not hasattr(out_grad, 'tmp_local_data'):
    out_grad.tmp_local_data = garray.empty_like(input_data)
  tmp_out_grad = out_grad.tmp_local_data

  output_y = grad.local_data.shape[r]
  output_x = grad.local_data.shape[c]

  image_y = input_data.shape[r]
  image_x = input_data.shape[c]

  garray.avgundo(
      input_data,
      grad.local_data,
      tmp_out_grad,
      pool_size, start, stride, output_y, output_x, image_y, image_x)

  out_grad.write(data = tmp_out_grad, area = input.tmp_local_area, propagate = propagate)

def rnorm(input, denom, output, channel, size, image_y, scaler, pow):
  r, c = ConvDataLayout.HEIGHT, ConvDataLayout.WIDTH
  state = get_state_from_distribution(output.slice_dim, conv = True)

  if state == disw_i:
    input.image_communicate(slice_dim = (r, c), stride = stride, filter_size = pool_size, output_area = output.local_area)
  elif state == disw_b:
    input.batch_communicate(input.rank, ConvDataLayout.BATCH)
  elif state == sisw:
    input.global_communicate()
  elif state == sidw:
    input.channel_communicate(input.rank, ConvDataLayout.CHANNEL)

  input_data = input.tmp_local_data

  if input.tmp_local_area == denom.local_area:
    denom.tmp_local_data = denom.local_data
    output.tmp_local_data = output.local_data
  else:
    # only happens when state == disw_i
    denom.tmp_local_data = garray.empty_like(input_data)
    output.tmp_local_data = garray.empty_like(input_data)
  tmp_denom_data = denom.tmp_local_data
  tmp_output_data = output.tmp_local_data

  image_y = input_data.shape[r]

  garray.rnorm(
      input_data,
      tmp_denom_data,
      tmp_output_data,
      channel, size, image_y, scaler, pow)

  if input.tmp_local_area != denom.local_area:
    output.write(area = input.tmp_local_area, data = tmp_output_data, propagate = False)
    denom.write(area = input.tmp_local_area, data = tmp_denom_data, propagate = False)

def rnormundo(grad, denom, input, output, out_grad, channel, size, image_y, scaler, pow):
  propagate = True
  r, c = ConvDataLayout.HEIGHT, ConvDataLayout.WIDTH
  state = get_state_from_distribution(grad.slice_dim, conv = True)
  if state == disw_i:
    if not hasattr(input, 'tmp_local_data'):
      input.image_communicate(slice_dim = (r, c), stride = 1, filter_size = size, output_area = grad.local_area)

    denom.image_communicate(slice_dim = (r, c), stride = 1, filter_size = size, output_area = grad.local_area)
    output.image_communicate(slice_dim = (r, c), stride = 1, filter_size = size, output_area = grad.local_area)
    grad.image_communicate(slice_dim = (r, c), stride = 1, filter_size = size, output_area = grad.local_area)
    if output.slice_dim == input.slice_dim: # previous layer has the same distribution disw_i
      propagate = False

  elif state == disw_b:
    if not hasattr(input, 'tmp_local_data'):
      input.batch_communicate(input.rank, ConvDataLayout.BATCH)
 
  elif state == sisw:
    if not hasattr(input, 'tmp_local_data'):
      input.global_communicate()
    propagate = False

  elif state == sidw:
    if not hasattr(input, 'tmp_local_data'):
      input.channel_communicate(input.rank, ConvDataLayout.CHANNEL)
    
  input_data = input.tmp_local_data

  if denom.local_area == input.tmp_local_area:
    denom_data = denom.local_data
    output_data = output.local_data
    grad_data = grad.local_data
  else:
    denom_data = denom.tmp_local_data
    output_data = output.tmp_local_data
    grad_data = grad.tmp_local_data
 
  if not hasattr(out_grad, 'tmp_local_data'):
    out_grad.tmp_local_data = garray.empty_like(input_data)
  tmp_out_grad = out_grad.tmp_local_data

  image_y = input.tmp_local_data.shape[r]

  garray.rnormundo(
      grad_data,
      denom_data,
      input_data,
      output_data,
      tmp_out_grad,
      channel, size, image_y, scaler, pow)
  
  if state == disw_i and propagate:
    tmp_out_grad = output.local_patch(tmp_out_grad)
  out_grad.write(data = tmp_out_grad, area = output.local_area, propagate = propagate)

def rnormcrossmap(input, denom, output, channel, size,image_y, scaler, pow, blocked):
  r, c = ConvDataLayout.HEIGHT, ConvDataLayout.WIDTH
  state = get_state_from_distribution(output.slice_dim, conv = True)
  if state == disw_i:
    input.image_communicate(slice_dim = (r, c), stride = 1, padding = 0, filter_size = 1, output_area = output.local_area)
  elif state == disw_b:
    input.batch_communicate(input.rank, ConvDataLayout.BATCH)
  elif state == sidw:
    input.channel_communicate(input.rank, ConvDataLayout.CHANNEL, padding = size / 2)
  elif state == sisw:
    input.global_communicate()
    
  input_data = input.tmp_local_data
  if input.tmp_local_area == denom.local_area:
    denom.tmp_local_data = denom.local_data
    output.tmp_local_data = output.local_data
  else:
    denom.tmp_local_data = garray.empty_like(input_data)
    output.tmp_local_data = garray.empty_like(input_data)
  tmp_denom_data = denom.tmp_local_data
  tmp_output_data = output.tmp_local_data

  image_y = input.local_data.shape[r]
  garray.rnormcrossmap(
      input.local_data,
      denom.local_data,
      output.local_data,
      channel, size, image_y, scaler, pow, blocked)

  if input.tmp_local_data != denom.local_data:
    output.write(area = denom.local_area, data = tmp_output_data, propagate = False)
    denom.write(area = denom.local_area, data = tmp_denom_data, propagate = False)

def rnormcrossmapundo(grad, denom, input, output, out_grad, channel, size, image_y, scaler, pow, blocked):
  propagate = True
  r, c = ConvDataLayout.HEIGHT, ConvDataLayout.WIDTH
  state = get_state_from_distribution(grad.slice_dim, conv = True)
  if state == disw_i:
    if not hasattr(input, 'tmp_local_data'):
      input.image_communicate(slice_dim = (r, c), stride = 1, filter_size = 1, output_area = grad.local_area)

  elif state == disw_b:
    if not hasattr(input, 'tmp_local_data'):
      input.batch_communicate(input.rank, ConvDataLayout.BATCH)
 
  elif state == sisw:
    if not hasattr(input, 'tmp_local_data'):
      input.global_communicate()
    propagate = False

  elif state == sidw:
    if not hasattr(input, 'tmp_local_data'):
      input.channel_communicate(input.rank, ConvDataLayout.CHANNEL, padding = size / 2)
    
    denom.channel_communicate(denom.rank, ConvDataLayout.CHANNEL, padding = size / 2)
    output.channel_communicate(output.rank, ConvDataLayout.CHANNEL, padding = size / 2)
    grad.channel_communicate(grad.rank, ConvDataLayout.CHANNEL, padding = size / 2)
    if output.slice_dim == input.slice_dim: # prievious layer has the some distribution sidw
      propagate = False
    
  input_data = input.tmp_local_data

  if denom.local_area == input.tmp_local_area:
    denom_data = denom.local_data
    output_data = output.local_data
    grad_data = grad.local_data
  else:
    denom_data = denom.tmp_local_data
    output_data = output.tmp_local_data
    grad_data = grad.tmp_local_data
 
  if not hasattr(out_grad, 'tmp_local_data'):
    out_grad.tmp_local_data = garray.empty_like(input_data)
  tmp_out_grad = out_grad.tmp_local_data

  image_y = input.local_data.shape[r]

  garray.rnormcrossmapundo(
      grad_data,
      denom_data,
      input_data,
      output_data,
      tmp_out_grad,
      channel, size, image_y, scaler, pow, blocked)

  if state == sidw and propagate:
    tmp_out_grad = output.local_path(tmp_out_grad)
  out_grad.write(data = tmp_out_grad, area = input.tmp_local_area, propagate = propagate)
