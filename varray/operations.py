import varray
import time
from distbase import util
from distbase.util import deprecated
import numpy as np
from .ndarray import VArray, DistMethod, zeros_like, WORLD, zeros, allocate_like, allocate, WORLD, barrier
from .area import Area
import garray
from distbase.state import *
from garray import ConvDataLayout, FCDataLayout, FilterLayout, WeightLayout

import sys

DEBUG = False

def copy_to(input, output):
  ''' used only by fc layer. When fprop, copy output to output, so the distributions of input and
  output are the same. When bprop, copy grad to grad. The input and output should share the same
  distribution. However, there is one exception. When copy to conv-related grad, the input is
  shared, while the output is distributed.'''
  if input.check_param(output):
    garray.copy_to(input.local_data, output.local_data)
  else:
    if output.global_unique or output.group_unique:
      assert not input.global_unique and not input.group_unique
      output.copy_from_global(input.local_data.reshape(output.shape))
    else:
      assert False

def partial_copy1(input, f, t):
  ''' partial copy last dimention '''
  shape = list(input.shape)
  shape[-1] = t-f
  rst = allocate(tuple(shape),
                 dtype = np.float32,
                 slice_dim = input.slice_dim,
                 slice_method = input.slice_method)
  old_shape = rst.local_shape
  rst.local_data = garray.partial_copy1(garray.reshape_last(input.local_data), f, t)
  rst.local_data = rst.local_data.reshape(old_shape)
  return rst

def bigger_than_scalar(input, scalar):
  garray.bigger_than_scalar(input.local_data, scalar)

def matrix_add(incr, grad ,alpha = 1.0, beta = 1.0):
  garray.matrix_add(incr.local_data, grad.local_data, alpha = alpha, beta = beta)

def sum(input, axis = None):
  ''' This function is used when getting the batch correctness '''
  if axis is None:
    # correctness
    return input.sum()

def argmax(input, axis):
  ''' This function is only used in logreg_cost function, in which the output is a synchronized
  varray or splitted across batch '''
  if input.group_unique == False and input.global_unique == False:
    return VArray(garray.argmax(input.local_data, axis = axis), context = input.context)

  assert input.group_slice_dim == FCDataLayout.BATCH and input.global_unique == False
  rst = VArray(shape = (1, input.shape[1]),
               group_slice_dim = FCDataLayout.BATCH,
               context = input.context)
  rst.local_data = garray.argmax(input.local_data, axis = axis) 
  return rst

def exp(input):
  c = allocate_like(input)
  garray.copy_to(input.local_data, c.local_data)
  return c

def iexp(input):
  garray.iexp(input.local_data)

def logreg_cost_col(output, label, cost):
  assert output.global_unique == False
  if output.group_unique == False:
    garray.logreg_cost_col(output.local_data, label, cost.local_data)
  else:
    garray.logreg_cost_col(output.local_data, label[cost.local_area.slice], cost.local_data)

def convert_from_data(input, output):
  ''' input has to be a GPUArray, and output has to be a VArray '''
  assert isinstance(input, garray.GPUArray)
  assert isinstance(output, VArray)

  global_output = garray.empty(shape = output.shape, dtype = np.float32)
  garray.convert_from_data(input, global_output)
  output.copy_from_global(global_output)

def fcforward(input, output, weight, bias, prev_conv):
  state = get_state_from_slice_dim(output.group_slice_dim, False, ConvDataLayout, FCDataLayout)
  if state == disw_b:
    input.batch_communicate(input.group_rank, ConvDataLayout.BATCH if prev_conv else FCDataLayout.BATCH)
  else:
    input.global_global_communicate()

  input_data = input.tmp_local_data
  if DEBUG:
    print '------ in fcforward ------'
    print 'state', state
    print 'input.shape', input_data.shape
    print 'weight.shape', weight.local_data.shape
    print 'output.shape', output.local_data.shape
  garray.matrixmult(weight.local_data, input_data, dest = output.local_data)
  garray.copy_to(output.local_data + bias.local_data , output.local_data)

def fcbackward(input, weight, grad, out_grad, weight_grad, bias_grad, prev_conv):
  grad_propagate = False
  weight_propagate = True
  state = get_state_from_slice_dim(grad.group_slice_dim, False, ConvDataLayout, FCDataLayout)
  
  if state == disw_b:
    if not hasattr(input, 'tmp_local_data'):
      input.batch_communicate(input.group_rank, FCDataLayout.BATCH)
  else:
    if not hasattr(input, 'tmp_local_data'):
      input.global_global_communicate()

  if state == sidw_f:
    grad_propagate = True
    if not hasattr(out_grad, 'tmp_local_data'):
      out_grad.tmp_local_data = garray.empty_like(input.tmp_local_data)
    tmp_out_grad = out_grad.tmp_local_data
  else:
    if input.tmp_local_area.cmp(out_grad.local_area):
      tmp_out_grad = out_grad.local_data
    else:
      if not hasattr(out_grad, 'tmp_local_data'):
        out_grad.tmp_local_data = garray.empty_like(input.tmp_local_data)
      tmp_out_grad = out_grad.tmp_local_data
      

  if state in [disw_b]:
    if not hasattr(weight_grad, 'tmp_local_data'):
      weight_grad.tmp_local_data = garray.empty_like(weight_grad.local_data)
    tmp_weight_grad = weight_grad.tmp_local_data
  else:
    tmp_weight_grad = weight_grad.local_data
    weight_propagate = False

  tmp_out_grad.fill(0)
  tmp_weight_grad.fill(0)

  if DEBUG:
    print '------ in fcbackward ------'
    print 'weight.shape', weight.local_data.shape
    print 'grad.shape', grad.local_data.shape
    print 'out_grad.shape', tmp_out_grad.shape
  garray.matrixmult(garray.transpose(weight.local_data), grad.local_data, dest = tmp_out_grad)
  garray.matrixmult(grad.local_data, garray.transpose(input.tmp_local_data), dest = tmp_weight_grad)
  
  weight_grad.write(area = weight_grad.local_area, data = tmp_weight_grad, propagate = weight_propagate)
  out_grad.write(area = input.tmp_local_area, data = tmp_out_grad, propagate = grad_propagate)

  garray.copy_to(garray.sum(grad.local_data, axis = 1), bias_grad.local_data)
  
def softmax(input, output):
  state = get_state_from_slice_dim(output.group_slice_dim, False, ConvDataLayout, FCDataLayout)
  if state == sisw:
    input.global_global_communicate()
    local_input = input.tmp_local_data
  else:
    local_input = input.local_data
  local_output = output.local_data
  max_rst = garray.max(local_input, axis = 0)
  garray.copy_to(local_input - max_rst, local_output)
  garray.iexp(local_output)
  sum_rst = garray.sum(local_output, axis = 0)
  garray.copy_to(local_output / sum_rst, local_output)

def softmax_bprop(output, label, out_grad):
  state = get_state_from_slice_dim(output.group_slice_dim, False, ConvDataLayout, FCDataLayout)
  if state == sisw:
    if not hasattr(out_grad, 'tmp_local_data'):
      out_grad.tmp_local_data = garray.empty_like(output.local_data)
    tmp_out_grad = out_grad.tmp_local_data
    tmp_out_grad.fill(0)
    if DEBUG:
      print '------ in softmax bprop ------'
      print 'output.shape', output.local_data.shape
      print 'label.shape', label.shape
      print 'out_grad.shape', tmp_out_grad.shape
    garray.softmax_bprop(output.local_data, label, tmp_out_grad)
    if DEBUG:
      print 'area', out_grad.local_area
      print 'data.shape', tmp_out_grad.shape
    out_grad.write(area = out_grad.global_area, data = tmp_out_grad, propagate = False)
  else:
    # if softmax is disw_b, so is fc
    idx = output.group_slice_dim
    f = output.local_area._from[idx]
    t = output.local_area._to[idx] + 1
    garray.softmax_bprop(output.local_data, label[0:1, f:t], out_grad.local_data)

def relu_activate(input, output, e):
  garray.relu_activate(input.local_data, output.local_data, e)

def relu_compute_grad(grad, output, out_grad, e):
  garray.relu_compute_grad(grad.local_data, output.local_data, out_grad.local_data, e)

def tanh_activate(input, output, a, b):
  garray.tanh_avtivate(input.local_data, output.local_data, a, b)

def tanh_compute_grad(grad, output, out_grad, a, b):
  garray.tanh_compute_grad(grad.local_data, output.local_data, out_grad.local_data, a, b)

def convolution(input, filter ,output, bias, image_y, output_y, output_x, padding, stride, channel, group):
  filter_size_index = FilterLayout.HEIGHT 
  r, c, ch = ConvDataLayout.HEIGHT, ConvDataLayout.WIDTH, ConvDataLayout.CHANNEL
  state = get_state_from_slice_dim(output.group_slice_dim, True, ConvDataLayout, FCDataLayout)

  if state == disw_i:
    input.image_communicate(slice_dim = (r, c),
                            padding = padding,
                            stride = stride,
                            filter_size = filter.local_shape[filter_size_index],
                            output_area = output.local_area)
    padding = 0
  elif state == disw_b:
    input.batch_communicate(input.group_rank, ConvDataLayout.BATCH)
  elif state == sidw or state == sisw:
    input.global_communicate()

  input_data = input.tmp_local_data
  image_y = input_data.shape[r]
  output_y = output.local_shape[r]
  output_x = output.local_shape[c]
  channel = input_data.shape[ch]

  if DEBUG:
    print '------ in convolution ------'
    print 'input.shape', input_data.shape
    print 'filter.shape', filter.local_data.shape
    print 'output.shape', output.local_data.shape
    print 'padding:',padding, 'stride:', stride, 'channel:', channel

  garray.convolution(
      input_data,
      filter.local_data,
      output.local_data,
      bias.local_data, 
      image_y, output_y, output_x, padding, stride, channel, group)

def bconvolution(input, grad, filter, out_grad, image_y, image_x, output_size, padding, stride, channel):
  real_padding = padding
  propagate = True
  r, c, ch = ConvDataLayout.HEIGHT, ConvDataLayout.WIDTH, ConvDataLayout.CHANNEL
  filter_size_index = FilterLayout.HEIGHT 
  state = get_state_from_slice_dim(grad.group_slice_dim, True, ConvDataLayout, FCDataLayout)

  if state == disw_i:
    if not hasattr(input, 'tmp_local_data'):
      input.image_communicate(slice_dim = (r, c),
                              padding = padding,
                              stride = stride,
                              filter_size = filter.local_shape[filter_size_index],
                              output_area = output.local_area)
    padding = 0
  elif state == disw_b:
    if not hasattr(input, 'tmp_local_data'):
      input.batch_communicate(input.group_rank, ConvDataLayout.BATCH)
  elif state == sidw or state == sisw:
    if not hasattr(input, 'tmp_local_data'):
      input.global_communicate()
    if state == sisw:
      propagate = False
    
  input_data = input.tmp_local_data

  if not hasattr(out_grad, 'tmp_local_data') or out_grad.tmp_local_data.shape != input_data.shape:
    out_grad.tmp_local_data = garray.empty_like(input_data)
  tmp_out_grad = out_grad.tmp_local_data

  image_y = input_data.shape[r]
  image_x = input_data.shape[c]
  output_size = grad.local_shape[r]
  channel = input_data.shape[ch]

  tmp_out_grad.fill(0)

  if DEBUG:
    print '------ in bconvolution ------'
    print 'input.shape', input_data.shape
    print 'grad.shape', grad.local_data.shape
    print 'filter.shape', filter.local_data.shape
    print 'out_grad.shape', tmp_out_grad.shape
    print 'padding:', padding, 'channel:', channel
  garray.bconvolution(
      input_data,
      grad.local_data,
      filter.local_data,
      tmp_out_grad,
      image_y, image_x, output_size, padding, stride, channel)
  
  if state == disw_i:
    tmp_out_grad = input.unpad(data = tmp_out_grad, 
                               padding = real_padding,
                               old_shape = tmp_out_grad.shape,
                               old_area = input.tmp_local_area,
                               slice_dim = (r, c),
                               debug = DEBUG)
  out_grad.write(area = input.tmp_local_area, data = tmp_out_grad, propagate = propagate, debug = DEBUG)

def wconvolution(input, grad, weight_grad, bias_grad, image_y, output_y, output_x, filter_size, padding, stride, channel):
  propagate = True
  filter_size_index = FilterLayout.HEIGHT
  r, c, ch = ConvDataLayout.HEIGHT, ConvDataLayout.WIDTH, ConvDataLayout.CHANNEL
  state = get_state_from_slice_dim(grad.group_slice_dim, True, ConvDataLayout, FCDataLayout)

  if state == disw_i:
    if not hasattr(input, 'tmp_local_data'):
      input.image_communicate(slice_dim = (r, c),
                              padding = padding,
                              stride = stride,
                              filter_size = filter.local_shape[filter_size_index],
                              output_area = output.local_area)
    padding = 0
  elif state == disw_b:
    if not hasattr(input, 'tmp_local_data'):
      input.batch_communicate(input.group_rank, ConvDataLayout.BATCH)
  elif state == sidw or state == sisw:
    if not hasattr(input, 'tmp_local_data'):
      input.global_communicate()
  
  input_data = input.tmp_local_data

  if state in [disw_i, disw_b]:
    #if not hasattr(weight_grad, 'tmp_local_data'):
    #  weight_grad.tmp_local_data = garray.GPUArray(weight_grad.shape, dtype = weight_grad.dtype)
    #tmp_weight_grad = weight_grad.tmp_local_data
    tmp_weight_grad = weight_grad.local_data
    if not hasattr(bias_grad, 'tmp_local_data'):
      bias_grad.tmp_local_data = garray.GPUArray(bias_grad.shape, dtype = bias_grad.dtype)
    tmp_bias_grad = bias_grad.tmp_local_data
  else:
    propagate = False
    tmp_weight_grad = weight_grad.local_data
    tmp_bias_grad = bias_grad.local_data

  image_y = input_data.shape[r]
  output_y = grad.local_shape[r]
  output_x = grad.local_shape[c]
  channel = input_data.shape[ch]
  
  tmp_weight_grad.fill(0)
  tmp_bias_grad.fill(0)

  if DEBUG:
    print '------ in wconvolution ------'
    print 'input.shape', input_data.shape
    print 'grad.shape', grad.local_data.shape
    print 'weight_grad.shape', tmp_weight_grad.shape
    print 'bias_grad.shape', tmp_bias_grad.shape
    print 'padding:', padding, 'channel:', channel
  garray.wconvolution(
      input_data,
      grad.local_data,
      tmp_weight_grad,
      tmp_bias_grad,
      image_y, output_y, output_x, filter_size, padding, stride, channel)
  
  if DEBUG:
    print 'area.shape', weight_grad.global_area.shape
    print 'data.shape', tmp_weight_grad.shape
    print 'propagate:', propagate
  #print 'rank %d, weight %f' % (input.global_rank, tmp_weight_grad.get()[0, 0, 0, 0])
  weight_grad.write(area = weight_grad.global_area, data = tmp_weight_grad, propagate = propagate)
  bias_grad.write(area = bias_grad.global_area, data = tmp_bias_grad, propagate = propagate)

def maxpool(input, output, channel, pool_size, start, stride, input_y, output_y, output_x):
  r, c, ch = ConvDataLayout.HEIGHT, ConvDataLayout.WIDTH, ConvDataLayout.CHANNEL
  state = get_state_from_slice_dim(output.group_slice_dim, True, ConvDataLayout, FCDataLayout)

  if state == disw_i:
    input.image_communicate(slice_dim = (r, c),
                            stride = stride,
                            filter_size = pool_size,
                            output_area = output.local_area)
  elif state == disw_b:
    input.batch_communicate(input.group_rank, ConvDataLayout.BATCH)
  elif state == sisw:
    input.global_communicate()
  elif state == sidw:
    input.channel_communicate(input.group_rank, ConvDataLayout.CHANNEL)

  input_data = input.tmp_local_data
  
  output_y = output.local_shape[r]
  output_x = output.local_shape[c]
  input_y = input_data.shape[r]
  channel = output.local_shape[ch]

  if DEBUG:
    print '------ in maxpool ------'
    print 'input.shape', input_data.shape
    print 'output.shape', output.local_data.shape
    print 'channel:', channel
  
  garray.maxpool(
      input_data,
      output.local_data,
      channel, pool_size, start, stride, input_y, output_y, output_x)

def maxundo(input, grad, output, out_grad, pool_size, start, stride, output_y, output_x, input_y):
  propagate = True
  r, c, ch = ConvDataLayout.HEIGHT, ConvDataLayout.WIDTH, ConvDataLayout.CHANNEL
  state = get_state_from_slice_dim(output.group_slice_dim, True, ConvDataLayout, FCDataLayout)

  if state == disw_i:
    if not hasattr(input, 'tmp_local_data'):
      input.image_communicate(slice_dim = (r, c),
                              stride = stride,
                              filter_size = pool_size,
                              output_area = grad.local_area)
  elif state == disw_b:
    if not hasattr(input, 'tmp_local_data'):
      input.batch_communicate(input.group_rank, ConvDataLayout.BATCH)
  elif state == sidw:
    if not hasattr(input, 'tmp_local_data'):
      input.channel_communicate(input.group_rank, ConvDataLayout.CHANNEL)
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
  channel = input_data.shape[ch]
  
  tmp_out_grad.fill(0)
  if DEBUG:
    print '------ in maxundo ------'
    print 'input.shape', input_data.shape
    print 'grad.shpae', grad.local_data.shape
    print 'output.shape', output.local_data.shape
    print 'out_grad.shape', tmp_out_grad.shape
    print 'channel:', channel
  garray.maxundo(
      input_data,
      grad.local_data,
      output.local_data,
      tmp_out_grad,
      pool_size, start, stride, output_y, output_x, input_y)

  if DEBUG:
    print 'outgrad area', input.tmp_local_area
    print 'data.shape', tmp_out_grad.shape
    print 'propagate', propagate
  out_grad.write(data = tmp_out_grad, area = input.tmp_local_area, propagate = propagate)

def avgpool(input, output, channel, pool_size, start, stride, input_y, output_y, output_x):
  r, c, ch = ConvDataLayout.HEIGHT, ConvDataLayout.WIDTH, ConvDataLayout.CHANNEL
  state = get_state_from_slice_dim(output.group_slice_dim, True, ConvDataLayout, FCDataLayout)
  
  if state == disw_i:
    input.image_communicate(slice_dim = (r, c),
                            stride = stride,
                            filter_size = pool_size,
                            output_area = output.local_area)
  elif state == disw_b:
    input.batch_communicate(input.group_rank, ConvDataLayout.BATCH)
  elif state == sisw:
    input.global_communicate()
  elif state == sidw:
    input.channel_communicate(input.group_rank, ConvDataLayout.CHANNEL)

  input_data = input.tmp_local_data

  output_y = output.local_shape[r]
  output_x = output.local_shape[c]
  input_y = input_data.shape[r]
  channel = input_data.shape[ch]

  if DEBUG:
    print '------ in avgpool ------'
    print 'input.shape', input_data.shape
    print 'output.shape', output.local_data.shape
    print 'channel:', channel

  garray.avgpool(
      input_data,
      output.local_data,
      channel, pool_size, start, stride, input_y, output_y, output_x)

def avgundo(input, grad, out_grad, pool_size, start, stride, output_y, output_x, image_y, image_x):
  propagate = True
  r, c, ch = ConvDataLayout.HEIGHT, ConvDataLayout.WIDTH, ConvDataLayout.CHANNEL
  state = get_state_from_slice_dim(grad.group_slice_dim, True, ConvDataLayout, FCDataLayout)

  if state == disw_i:
    if not hasattr(input, 'tmp_local_data'):
      input.image_communicate(slice_dim = (r, c),
                              stride = stride,
                              filter_size = pool_size,
                              output_area = grad.local_area)
  elif state == disw_b:
    if not hasattr(input, 'tmp_local_data'):
      input.batch_communicate(input.group_rank, ConvDataLayout.BATCH)
  elif state == sidw:
    if not hasattr(input, 'tmp_local_data'):
      input.channle_communicate(input.group_rank, ConvDataLayout.CHANNEL)
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
  channel = input_data.shape[ch]

  tmp_out_grad.fill(0)

  if DEBUG:
    print '------ in avgundo ------'
    print 'input.shape', input_data.shape
    print 'grad.shape', grad.local_data.shape
    print 'out_grad.shape', tmp_out_grad.shape
    print 'channel:', channel
  garray.avgundo(
      input_data,
      grad.local_data,
      tmp_out_grad,
      pool_size, start, stride, output_y, output_x, image_y, image_x)

  if DEBUG:
    print 'outgrad area', input.tmp_local_area
    print 'data.shape', tmp_out_grad.shape
    print 'propagate', propagate
  out_grad.write(data = tmp_out_grad, area = input.tmp_local_area, propagate = propagate)

def rnorm(input, denom, output, channel, size, image_y, scalar, pow):
  r, c, ch = ConvDataLayout.HEIGHT, ConvDataLayout.WIDTH, ConvDataLayout.CHANNEL
  state = get_state_from_slice_dim(output.group_slice_dim, True, ConvDataLayout, FCDataLayout)

  if state == disw_i:
    input.image_communicate(slice_dim = (r, c),
                            stride = 1,
                            filter_size = size,
                            output_area = output.local_area)
  elif state == disw_b:
    input.batch_communicate(input.group_rank, ConvDataLayout.BATCH)
  elif state == sisw:
    input.global_communicate()
  elif state == sidw:
    input.channel_communicate(input.group_rank, ConvDataLayout.CHANNEL)

  input_data = input.tmp_local_data

  if input.tmp_local_area.cmp(denom.local_area):
    denom.tmp_local_data = denom.local_data
    output.tmp_local_data = output.local_data
  else:
    # only happens when state == disw_i
    denom.tmp_local_data = garray.empty_like(input_data)
    output.tmp_local_data = garray.empty_like(input_data)
  tmp_denom_data = denom.tmp_local_data
  tmp_output_data = output.tmp_local_data

  image_y = input_data.shape[r]
  channel = input_data.shape[ch]
  
  if DEBUG:
    print '------ in rnorm ------'
    print 'input.shape', input_data.shape
    print 'denom.shape', tmp_denom_data.shape
    print 'output.shape', tmp_output_data.shape
    print 'channel:', channel
  garray.rnorm(
      input_data,
      tmp_denom_data,
      tmp_output_data,
      channel, size, image_y, scalar, pow)

  if input.tmp_local_area != denom.local_area:
    output.write(area = input.tmp_local_area, data = tmp_output_data, propagate = False)
    denom.write(area = input.tmp_local_area, data = tmp_denom_data, propagate = False)

def rnormundo(grad, denom, input, output, out_grad, channel, size, image_y, scalar, pow):
  propagate = True
  r, c, ch = ConvDataLayout.HEIGHT, ConvDataLayout.WIDTH, ConvDataLayout.CHANNEL
  out_grad.fill(0)
  state = get_state_from_slice_dim(grad.group_slice_dim, True, ConvDataLayout, FCDataLayout)
  if state == disw_i:
    if not hasattr(input, 'tmp_local_data'):
      input.image_communicate(slice_dim = (r, c),
                              stride = 1,
                              filter_size = size,
                              output_area = grad.local_area)

    denom.image_communicate(slice_dim = (r, c),
                            stride = 1,
                            filter_size = size,
                            output_area = grad.local_area)
    output.image_communicate(slice_dim = (r, c),
                             stride = 1,
                             filter_size = size,
                             output_area = grad.local_area)
    grad.image_communicate(slice_dim = (r, c),
                           stride = 1,
                           filter_size = size,
                           output_area = grad.local_area)
    if output.group_slice_dim == input.group_slice_dim: # previous layer has the same distribution disw_i
      propagate = False

  elif state == disw_b:
    if not hasattr(input, 'tmp_local_data'):
      input.batch_communicate(input.group_rank, ConvDataLayout.BATCH)
 
  elif state == sisw:
    if not hasattr(input, 'tmp_local_data'):
      input.global_communicate()
    propagate = False

  elif state == sidw:
    if not hasattr(input, 'tmp_local_data'):
      input.channel_communicate(input.group_rank, ConvDataLayout.CHANNEL)
    
  input_data = input.tmp_local_data

  if denom.local_area.cmp(input.tmp_local_area):
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
  channel = input.tmp_local_data.shape[ch] 

  tmp_out_grad.fill(0)
  
  if DEBUG:
    print '------ in rnormundo ------'
    print 'grad.shape', grad_data.shape
    print 'denom.shape', denom_data.shape
    print 'input.shape', input_data.shape
    print 'output.shape', output_data.shape
    print 'out_grad.shape', tmp_out_grad.shape
    print 'channel:', channel

  garray.rnormundo(
      grad_data,
      denom_data,
      input_data,
      output_data,
      tmp_out_grad,
      channel, size, image_y, scalar, pow)
  
  if state == disw_i and propagate:
    tmp_out_grad = output.local_patch(tmp_out_grad)
  out_grad.write(data = tmp_out_grad, area = output.local_area, propagate = propagate)

def rnormcrossmap(input, denom, output, channel, size,image_y, scalar, pow, blocked):
  r, c, ch = ConvDataLayout.HEIGHT, ConvDataLayout.WIDTH, ConvDataLayout.CHANNEL
  state = get_state_from_slice_dim(output.group_slice_dim, True, ConvDataLayout, FCDataLayout)
  if state == disw_i:
    input.image_communicate(slice_dim = (r, c),
                            stride = 1,
                            padding = 0,
                            filter_size = 0,
                            output_area = output.local_area)
  elif state == disw_b:
    input.batch_communicate(input.group_rank, ConvDataLayout.BATCH)
  elif state == sidw:
    input.channel_communicate(input.group_rank, ConvDataLayout.CHANNEL, padding = size / 2)
  elif state == sisw:
    input.global_communicate()
    
  input_data = input.tmp_local_data
  if input.tmp_local_area.cmp(denom.local_area):
    denom.tmp_local_data = denom.local_data
    output.tmp_local_data = output.local_data
  else:
    denom.tmp_local_data = garray.empty_like(input_data)
    output.tmp_local_data = garray.empty_like(input_data)
  tmp_denom_data = denom.tmp_local_data
  tmp_output_data = output.tmp_local_data

  image_y = input.local_data.shape[r]
  channel = input_data.shape[ch]
  
  if DEBUG:
    print '------ in rnormcrossmap ------'
    print 'input.shape', input_data.shape
    print 'denom.shape', tmp_denom_data.shape
    print 'output.shape', tmp_output_data.shape
    print 'channel:', channel

  garray.rnormcrossmap(
      input_data,
      tmp_denom_data,
      tmp_output_data,
      channel, size, image_y, scalar, pow, blocked)

  if input.tmp_local_area.cmp(denom.local_area):
    output.write(area = denom.local_area, data = tmp_output_data, propagate = False)
    denom.write(area = denom.local_area, data = tmp_denom_data, propagate = False)

def rnormcrossmapundo(grad, denom, input, output, out_grad, channel, size, image_y, scalar, pow, blocked):
  propagate = True
  r, c,ch = ConvDataLayout.HEIGHT, ConvDataLayout.WIDTH, ConvDataLayout.CHANNEL
  state = get_state_from_slice_dim(grad.group_slice_dim, True, ConvDataLayout, FCDataLayout)
  if state == disw_i:
    if not hasattr(input, 'tmp_local_data'):
      input.image_communicate(slice_dim = (r, c),
                              stride = 1,
                              filter_size = 0,
                              output_area = grad.local_area)

  elif state == disw_b:
    if not hasattr(input, 'tmp_local_data'):
      input.batch_communicate(input.group_rank, ConvDataLayout.BATCH)
 
  elif state == sisw:
    if not hasattr(input, 'tmp_local_data'):
      input.global_communicate()
    propagate = False

  elif state == sidw:
    if not hasattr(input, 'tmp_local_data'):
      input.channel_communicate(input.group_rank, ConvDataLayout.CHANNEL, padding = size / 2)
    
    denom.channel_communicate(denom.group_rank, ConvDataLayout.CHANNEL, padding = size / 2)
    output.channel_communicate(output.group_rank, ConvDataLayout.CHANNEL, padding = size / 2)
    grad.channel_communicate(grad.group_rank, ConvDataLayout.CHANNEL, padding = size / 2)
    if output.group_slice_dim == input.group_slice_dim: # prievious layer has the some distribution sidw
      propagate = False
    
  input_data = input.tmp_local_data

  if denom.local_area.cmp(input.tmp_local_area):
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
  channel = input_data.shape[ch]

  tmp_out_grad.fill(0)
  
  if DEBUG:
    print '------ in rnormcrossmapundo ------'
    print 'grad.shape', grad_data.shape
    print 'denom.shape', denom_data.shape
    print 'input.shape', input_data.shape
    print 'output.shape', output_data.shape
    print 'out_grad.shape', tmp_out_grad.shape
    print 'channel:', channel

  garray.rnormcrossmapundo(
      grad_data,
      denom_data,
      input_data,
      output_data,
      tmp_out_grad,
      channel, size, image_y, scalar, pow, blocked)

  if state == sidw and propagate:
    tmp_out_grad = output.local_path(tmp_out_grad)
  out_grad.write(data = tmp_out_grad, area = input.tmp_local_area, propagate = propagate)
