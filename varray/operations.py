import varray
import time
from distbase import util
from distbase.util import deprecated
from distbase.monitor import MONITOR
import numpy as np
from .ndarray import VArray, zeros_like, WORLD, zeros, allocate_like, allocate, WORLD, barrier, INNER
from .area import Area
import garray
from distbase.state import *
from garray import ConvDataLayout, FCDataLayout, FilterLayout, WeightLayout, driver

import sys

DEBUG = False

def copy_to(input, output):
  ''' used only by fc layer. When fprop, copy output to output, so the distributions of input and
  output are the same. When bprop, copy grad to grad. The input and output should share the same
  distribution. However, there is one exception. When copy to conv-related grad, the input is
  shared, while the output is distributed.'''
  if input.check_param(output):
    garray.copy_to(input.DATA, output.DATA)
  else:
    if output.global_unique or output.group_unique:
      assert not input.global_unique and not input.group_unique
      output.copy_from_global(input.DATA.reshape(output.shape))
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
  rst.DATA = garray.partial_copy1(garray.reshape_last(input.DATA), f, t)
  rst.DATA = rst.DATA.reshape(old_shape)
  return rst

def bigger_than_scalar(input, scalar):
  garray.bigger_than_scalar(input.DATA, scalar)

def matrix_add(incr, grad ,alpha = 1.0, beta = 1.0):
  garray.matrix_add(incr.DATA, grad.DATA, alpha = alpha, beta = beta)

def sum(input, axis = None):
  ''' This function is used when getting the batch correctness '''
  if axis is None:
    # correctness
    return input.sum()

def argmax(input, axis):
  ''' This function is only used in logreg_cost function, in which the output is a synchronized
  varray or splitted across batch '''
  if input.group_unique == False and input.global_unique == False:
    return VArray(garray.argmax(input.DATA, axis = axis), context = input.context)

  assert input.group_slice_dim == FCDataLayout.BATCH and input.global_unique == False
  rst = VArray(shape = (1, input.shape[1]),
               group_slice_dim = FCDataLayout.BATCH,
               context = input.context)
  rst.DATA = garray.argmax(input.DATA, axis = axis)
  return rst

def exp(input):
  c = allocate_like(input)
  garray.copy_to(input.DATA, c.DATA)
  return c

def iexp(input):
  garray.iexp(input.DATA)

def logreg_cost_col(output, label, cost):
  assert output.global_unique == False
  if output.group_unique == False:
    garray.logreg_cost_col(output.DATA, label, cost.DATA)
  else:
    garray.logreg_cost_col(output.DATA, label[cost.local_area.slice], cost.DATA)

def convert_from_data(input, output):
  ''' input has to be a GPUArray, and output has to be a VArray '''
  _ = time.time()
  #assert isinstance(input, VArray)
  assert isinstance(output, VArray)

  if isinstance(input, garray.GPUArray):
    output.copy_from_global(input)
    return

  if input.check_param(output):
    garray.convert_from_data(input.DATA, output.DATA)
  else:
    if not input.global_unique and not input.group_unique:
      output.copy_from_global(input.DATA)
    else:
      if input.global_unique != output.global_unique:
        # must regroup the input
        input.regroup_like(output)
      if input.group_slice_dim == output.group_slice_dim:
        garray.convert_from_data(input.DATA, output.DATA)
      else:
        group_output = garray.empty(shape = input.group_area.shape, dtype = np.float32)
        input.group_gather()
        garray.convert_from_data(input.DATA, group_output)
        output.copy_from_group(group_output)

  driver.Context.synchronize()
  if not INNER: MONITOR.add_comm(time.time() - _)

def convert_from_backend(weight, backend):
  return garray.convert_from_backend(weight, backend)

def fcforward(input, output, weight, bias, prev_conv):
  state = get_state_from_slice_dim(output.group_slice_dim, False, ConvDataLayout, FCDataLayout)
  _ = time.time()
  if state == disw_b:
    input.batch_communicate(input.group_rank, ConvDataLayout.BATCH if prev_conv else FCDataLayout.BATCH)
  else:
    input.global_global_communicate()
  if not INNER: MONITOR.add_comm(time.time() - _)

  _ = time.time()
  input_data = input.tmp_local_data
  garray.matrixmult(weight.DATA, input_data, dest = output.DATA)
  garray.copy_to(output.DATA + bias.DATA , output.DATA)
  driver.Context.synchronize()
  MONITOR.add_fprop(time.time() - _)

def fcbackward(input, weight, grad, out_grad, weight_grad, bias_grad, prev_conv):
  grad_propagate = False
  weight_propagate = True
  state = get_state_from_slice_dim(grad.group_slice_dim, False, ConvDataLayout, FCDataLayout)

  _ = time.time()
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
      tmp_out_grad = out_grad.DATA
    else:
      if not hasattr(out_grad, 'tmp_local_data'):
        out_grad.tmp_local_data = garray.empty_like(input.tmp_local_data)
      tmp_out_grad = out_grad.tmp_local_data

  if not INNER: MONITOR.add_comm(time.time() - _)

  if state in [disw_b]:
    if not hasattr(weight_grad, 'tmp_local_data'):
      weight_grad.tmp_local_data = garray.empty_like(weight_grad.DATA)
    tmp_weight_grad = weight_grad.tmp_local_data
  else:
    tmp_weight_grad = weight_grad.DATA
    weight_propagate = False

  _ = time.time()
  tmp_out_grad.fill(0)
  tmp_weight_grad.fill(0)
  garray.matrixmult(garray.transpose(weight.DATA), grad.DATA, dest = tmp_out_grad)
  driver.Context.synchronize()
  MONITOR.add_bprop(time.time() - _)
  _ = time.time()
  garray.matrixmult(grad.DATA, garray.transpose(input.tmp_local_data), dest = tmp_weight_grad)
  garray.copy_to(garray.sum(grad.DATA, axis = 1), bias_grad.DATA)
  driver.Context.synchronize()
  MONITOR.add_wprop(time.time() - _)
  _ = time.time()
  weight_grad.write(area = weight_grad.local_area, data = tmp_weight_grad, propagate = weight_propagate)
  out_grad.write(area = input.tmp_local_area, data = tmp_out_grad, propagate = grad_propagate)

  if not INNER: MONITOR.add_comm(time.time() - _)

def softmax(input, output):
  state = get_state_from_slice_dim(output.group_slice_dim, False, ConvDataLayout, FCDataLayout)
  if state == sisw:
    _ = time.time()
    input.global_global_communicate()
    if not INNER: MONITOR.add_comm(time.time() - _)
    local_input = input.tmp_local_data
  else:
    local_input = input.DATA
  _ = time.time()
  local_output = output.DATA
  max_rst = garray.max(local_input, axis = 0)
  garray.copy_to(local_input - max_rst, local_output)
  garray.iexp(local_output)
  sum_rst = garray.sum(local_output, axis = 0)
  garray.copy_to(local_output / sum_rst, local_output)
  MONITOR.add_fprop(time.time() - _)

def softmax_bprop(output, label, out_grad):
  state = get_state_from_slice_dim(output.group_slice_dim, False, ConvDataLayout, FCDataLayout)
  if state == sisw:
    _ = time.time()
    if not hasattr(out_grad, 'tmp_local_data'):
      out_grad.tmp_local_data = garray.empty_like(output.DATA)
    tmp_out_grad = out_grad.tmp_local_data
    tmp_out_grad.fill(0)
    garray.softmax_bprop(output.DATA, label, tmp_out_grad)
    MONITOR.add_bprop(time.time() - _)
    _ = time.time()
    out_grad.write(area = out_grad.global_area, data = tmp_out_grad, propagate = False)
    if not INNER: MONITOR.add_comm(time.time() - _)
  else:
    # if softmax is disw_b, so is fc
    _ = time.time()
    idx = output.group_slice_dim
    f = output.local_area._from[idx]
    t = output.local_area._to[idx] + 1
    garray.softmax_bprop(output.DATA, label[0:1, f:t], out_grad.DATA)
    MONITOR.add_bprop(time.time() - _)

def relu_activate(input, output, e):
  garray.relu_activate(input.DATA, output.DATA, e)

def relu_compute_grad(grad, output, out_grad, e):
  garray.relu_compute_grad(grad.DATA, output.DATA, out_grad.DATA, e)

def tanh_activate(input, output, a, b):
  garray.tanh_avtivate(input.DATA, output.DATA, a, b)

def tanh_compute_grad(grad, output, out_grad, a, b):
  garray.tanh_compute_grad(grad.DATA, output.DATA, out_grad.DATA, a, b)

def convolution(input, filter ,output, bias, padding, stride):
  filter_size_index = FilterLayout.HEIGHT
  r, c, ch = ConvDataLayout.HEIGHT, ConvDataLayout.WIDTH, ConvDataLayout.CHANNEL
  state = get_state_from_slice_dim(output.group_slice_dim, True, ConvDataLayout, FCDataLayout)

  _ = time.time()
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

  if not INNER: MONITOR.add_comm(time.time() - _)

  _ = time.time()
  garray.convolution(input_data, filter.DATA, output.DATA, bias.DATA, padding, stride)
  MONITOR.add_fprop(time.time() - _)

def bconvolution(input, grad, filter, out_grad, padding, stride):
  real_padding = padding
  propagate = True
  r, c, ch = ConvDataLayout.HEIGHT, ConvDataLayout.WIDTH, ConvDataLayout.CHANNEL
  filter_size_index = FilterLayout.HEIGHT
  state = get_state_from_slice_dim(grad.group_slice_dim, True, ConvDataLayout, FCDataLayout)

  _ = time.time()
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

  if not INNER: MONITOR.add_comm(time.time() - _)

  input_data = input.tmp_local_data

  if not hasattr(out_grad, 'tmp_local_data') or out_grad.tmp_local_data.shape != input_data.shape:
    out_grad.tmp_local_data = garray.empty_like(input_data)
  tmp_out_grad = out_grad.tmp_local_data

  _ = time.time()
  tmp_out_grad.fill(0)

  garray.bconvolution( input_data, grad.DATA, filter.DATA, tmp_out_grad, padding, stride)

  MONITOR.add_bprop(time.time() - _)

  _ = time.time()
  if state == disw_i:
    tmp_out_grad = input.unpad(data = tmp_out_grad,
                               padding = real_padding,
                               old_shape = tmp_out_grad.shape,
                               old_area = input.tmp_local_area,
                               slice_dim = (r, c),
                               debug = DEBUG)
  out_grad.write(area = input.tmp_local_area, data = tmp_out_grad, propagate = propagate, debug = DEBUG)
  if not INNER: MONITOR.add_comm(time.time() - _)

def wconvolution(input, grad, weight_grad, bias_grad, padding, stride, *args):
  propagate = True
  filter_size_index = FilterLayout.HEIGHT
  r, c, ch = ConvDataLayout.HEIGHT, ConvDataLayout.WIDTH, ConvDataLayout.CHANNEL
  state = get_state_from_slice_dim(grad.group_slice_dim, True, ConvDataLayout, FCDataLayout)

  _ = time.time()

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

  if not INNER: MONITOR.add_comm(time.time() - _)

  if state in [disw_i, disw_b]:
    tmp_weight_grad = weight_grad.DATA
    if not hasattr(bias_grad, 'tmp_local_data'):
      bias_grad.tmp_local_data = garray.GPUArray(bias_grad.shape, dtype = bias_grad.dtype)
    tmp_bias_grad = bias_grad.tmp_local_data
  else:
    propagate = False
    tmp_weight_grad = weight_grad.DATA
    tmp_bias_grad = bias_grad.DATA

  _ = time.time()

  tmp_weight_grad.fill(0)
  tmp_bias_grad.fill(0)

  garray.wconvolution( input_data, grad.DATA, tmp_weight_grad, tmp_bias_grad,padding, stride, chann*args)

  MONITOR.add_wprop(time.time() - _)
  _ = time.time()
  weight_grad.write(area = weight_grad.global_area, data = tmp_weight_grad, propagate = propagate)
  bias_grad.write(area = bias_grad.global_area, data = tmp_bias_grad, propagate = propagate)
  if not INNER: MONITOR.add_comm(time.time() - _)

def maxpool(input, output, pool_size, start, stride):
  r, c, ch = ConvDataLayout.HEIGHT, ConvDataLayout.WIDTH, ConvDataLayout.CHANNEL
  state = get_state_from_slice_dim(output.group_slice_dim, True, ConvDataLayout, FCDataLayout)

  _ = time.time()
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

  if not INNER: MONITOR.add_comm(time.time() - _)
  input_data = input.tmp_local_data

  _ = time.time()
  garray.maxpool(input_data, output.DATA, pool_size, start, stride)

  MONITOR.add_fprop(time.time() - _)

def maxundo(input, grad, output, out_grad, pool_size, start, stride):
  propagate = True
  r, c, ch = ConvDataLayout.HEIGHT, ConvDataLayout.WIDTH, ConvDataLayout.CHANNEL
  state = get_state_from_slice_dim(output.group_slice_dim, True, ConvDataLayout, FCDataLayout)

  _ = time.time()
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

  if not INNER: MONITOR.add_comm(time.time() - _)

  input_data = input.tmp_local_data

  if input_data.shape != out_grad.DATA.shape:
    if not hasattr(out_grad, 'tmp_local_data'):
      out_grad.tmp_local_data = garray.empty_like(input_data)
  else:
    out_grad.tmp_local_data = out_grad.DATA
  tmp_out_grad = out_grad.tmp_local_data

  _ = time.time()
  tmp_out_grad.fill(0)
  garray.maxundo(input_data, grad.DATA, output.DATA, tmp_out_grad, pool_size, start, stride)

  MONITOR.add_bprop(time.time() - _)

  _ = time.time()
  out_grad.write(data = tmp_out_grad, area = input.tmp_local_area, propagate = propagate)
  if not INNER: MONITOR.add_comm(time.time() - _)

def avgpool(input, output, pool_size, start, stride):
  r, c, ch = ConvDataLayout.HEIGHT, ConvDataLayout.WIDTH, ConvDataLayout.CHANNEL
  state = get_state_from_slice_dim(output.group_slice_dim, True, ConvDataLayout, FCDataLayout)

  _ = time.time()
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

  if not INNER: MONITOR.add_comm(time.time() - _)
  input_data = input.tmp_local_data

  _ = time.time()
  garray.avgpool(input_data, output.DATA, pool_size, start, stride)

  MONITOR.add_fprop(time.time() - _)

def avgundo(input, grad, out_grad, pool_size, start, stride)
  propagate = True
  r, c, ch = ConvDataLayout.HEIGHT, ConvDataLayout.WIDTH, ConvDataLayout.CHANNEL
  state = get_state_from_slice_dim(grad.group_slice_dim, True, ConvDataLayout, FCDataLayout)

  _ = time.time()
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

  if not INNER: MONITOR.add_comm(time.time() - _)
  input_data = input.tmp_local_data

  if not hasattr(out_grad, 'tmp_local_data'):
    out_grad.tmp_local_data = garray.empty_like(input_data)
  tmp_out_grad = out_grad.tmp_local_data

  _ = time.time()
  tmp_out_grad.fill(0)
  garray.avgundo( input_data, grad.DATA, tmp_out_grad, pool_size, start, stride)

  MONITOR.add_bprop(time.time() - _)
  _ = time.time()
  out_grad.write(data = tmp_out_grad, area = input.tmp_local_area, propagate = propagate)
  if not INNER: MONITOR.add_comm(time.time() - _)

def rnorm(input, denom, output, size, scalar, pow):
  r, c, ch = ConvDataLayout.HEIGHT, ConvDataLayout.WIDTH, ConvDataLayout.CHANNEL
  state = get_state_from_slice_dim(output.group_slice_dim, True, ConvDataLayout, FCDataLayout)

  _ = time.time()
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

  if not INNER: MONITOR.add_comm(time.time() - _)
  input_data = input.tmp_local_data

  if input.tmp_local_area.cmp(denom.local_area):
    denom.tmp_local_data = denom.DATA
    output.tmp_local_data = output.DATA
  else:
    # only happens when state == disw_i
    denom.tmp_local_data = garray.empty_like(input_data)
    output.tmp_local_data = garray.empty_like(input_data)
  tmp_denom_data = denom.tmp_local_data
  tmp_output_data = output.tmp_local_data

  _ = time.time()
  garray.rnorm(input_data, tmp_denom_data, tmp_output_data, size, scalar, pow)

  MONITOR.add_fprop(time.time() - _)

  _ = time.time()
  if input.tmp_local_area != denom.local_area:
    output.write(area = input.tmp_local_area, data = tmp_output_data, propagate = False)
    denom.write(area = input.tmp_local_area, data = tmp_denom_data, propagate = False)
  if not INNER: MONITOR.add_comm(time.time() - _)

def rnormundo(grad, denom, input, output, out_grad, size, scalar, pow):
  propagate = True
  r, c, ch = ConvDataLayout.HEIGHT, ConvDataLayout.WIDTH, ConvDataLayout.CHANNEL
  out_grad.fill(0)
  state = get_state_from_slice_dim(grad.group_slice_dim, True, ConvDataLayout, FCDataLayout)

  _ = time.time()
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

  if not INNER: MONITOR.add_comm(time.time() - _)
  input_data = input.tmp_local_data

  if denom.local_area.cmp(input.tmp_local_area):
    denom_data = denom.DATA
    output_data = output.DATA
    grad_data = grad.DATA
  else:
    denom_data = denom.tmp_local_data
    output_data = output.tmp_local_data
    grad_data = grad.tmp_local_data

  if not hasattr(out_grad, 'tmp_local_data'):
    out_grad.tmp_local_data = garray.empty_like(input_data)
  tmp_out_grad = out_grad.tmp_local_data

  _ = time.time()
  tmp_out_grad.fill(0)

  garray.rnormundo( grad_data, denom_data, input_data, output_data, tmp_out_grad, size,  scalar, pow)

  MONITOR.add_bprop(time.time() - _)
  _ = time.time()
  if state == disw_i and propagate:
    tmp_out_grad = output.local_patch(tmp_out_grad)
  out_grad.write(data = tmp_out_grad, area = output.local_area, propagate = propagate)
  if not INNER: MONITOR.add_comm(time.time() - _)

def rnormcrossmap(input, denom, output, size, scalar, pow, blocked):
  r, c, ch = ConvDataLayout.HEIGHT, ConvDataLayout.WIDTH, ConvDataLayout.CHANNEL
  state = get_state_from_slice_dim(output.group_slice_dim, True, ConvDataLayout, FCDataLayout)

  _ = time.time()
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
  if not INNER: MONITOR.add_comm(time.time() - _)

  input_data = input.tmp_local_data
  if input.tmp_local_area.cmp(denom.local_area):
    denom.tmp_local_data = denom.DATA
    output.tmp_local_data = output.DATA
  else:
    denom.tmp_local_data = garray.empty_like(input_data)
    output.tmp_local_data = garray.empty_like(input_data)
  tmp_denom_data = denom.tmp_local_data
  tmp_output_data = output.tmp_local_data

  _ = time.time()
  garray.rnormcrossmap( input_data, tmp_denom_data, tmp_output_data, size,  scalar, pow, blocked)
  MONITOR.add_fprop(time.time() - _)
  _ = time.time()

  if not input.tmp_local_area.cmp(denom.local_area):
    output.write(area = denom.local_area, data = tmp_output_data, propagate = False)
    denom.write(area = denom.local_area, data = tmp_denom_data, propagate = False)
  if not INNER: MONITOR.add_comm(time.time() - _)

def rnormcrossmapundo(grad, denom, input, output, out_grad, size,  scalar, pow, blocked):
  propagate = True
  r, c,ch = ConvDataLayout.HEIGHT, ConvDataLayout.WIDTH, ConvDataLayout.CHANNEL
  state = get_state_from_slice_dim(grad.group_slice_dim, True, ConvDataLayout, FCDataLayout)

  _ = time.time()
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

  if not INNER: MONITOR.add_comm(time.time() - _)
  input_data = input.tmp_local_data

  if denom.local_area.cmp(input.tmp_local_area):
    denom_data = denom.DATA
    output_data = output.DATA
    grad_data = grad.DATA
  else:
    denom_data = denom.tmp_local_data
    output_data = output.tmp_local_data
    grad_data = grad.tmp_local_data

  if input_data.shape != out_grad.DATA.shape:
    if not hasattr(out_grad, 'tmp_local_data'):
      out_grad.tmp_local_data = garray.empty_like(input_data)
    tmp_out_grad = out_grad.tmp_local_data
  else:
    tmp_out_grad = out_grad.DATA

  _ = time.time()
  tmp_out_grad.fill(0)

  garray.rnormcrossmapundo( grad_data, denom_data, input_data, output_data, tmp_out_grad, size,  scalar, pow, blocked) 
  MONITOR.add_bprop(time.time() - _)
  _ = time.time()
  if state == sidw and propagate:
    tmp_out_grad = output.local_path(tmp_out_grad)
  if not denom.local_area.cmp(input.tmp_local_area):
    out_grad.write(data = tmp_out_grad, area = input.tmp_local_area, propagate = propagate)
  if not INNER: MONITOR.add_comm(time.time() - _)
