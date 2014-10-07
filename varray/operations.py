import varray
import time
from distbase import util
from distbase.util import deprecated
from distbase.monitor import MONITOR
import numpy as np
from .ndarray import VArray, zeros_like, WORLD, zeros, allocate_like, allocate, WORLD, barrier, INNER, rank
from .area import Area
import garray
from garray import ConvDataLayout, FCDataLayout, FilterLayout, WeightLayout, driver

import sys

DEBUG = False

def copy_to(input, output):
  '''
  Copy the content of input to output
  @param input:source
  @param output:destination

  This function only used by fc layer.
    - When fprop, copy output*mask to output, the distribution of input and output must be the same
    - When bprop, copy grad*mask to grad
  '''
  if input.check_param(output):
    garray.copy_to(input.DATA, output.DATA)
  else:
    assert False

def bigger_than_scalar(input, scalar):
  '''
  Same as bigger_than_scalar in garray
  '''
  garray.bigger_than_scalar(input.DATA, scalar)

def matrix_add(incr, grad ,alpha = 1.0, beta = 1.0):
  '''
  Same as matrix_add in garray
  @param incr and @param grad must have the same distribution
  '''
  if incr.check_param(grad):
    garray.matrix_add(incr.DATA, grad.DATA, alpha = alpha, beta = beta)
  else:
    assert False

def sum(input, axis = None):
  '''
  Get the sum of an array in any given axis
  @param input:Array object
  @param axis[None]:Axis that needs to compute

  This function is only used when getting the batch correctness, in which case axis would be None
  and input will be both global and group shared.
  '''
  if axis is None and not input.global_unique and not input.group_unique:
    return garray.sum(input.DATA)
  else:
    assert False

def argmax(input, axis):
  '''
  Get the max index of an array in any given axis
  @param input:Array object
  @param axis:Axis that needs to compute

  This function is only used in logreg_cost function, in which case the output is a synchronized
  varray, because softmax layer has to apply replica parallel or fake parallel
  '''
  if input.group_unique == False and input.global_unique == False:
    return VArray(garray.argmax(input.DATA, axis = axis), context = input.context)

def iexp(input):
  '''
  Same as iexp in array
  '''
  garray.iexp(input.DATA)

def logreg_cost_col(output, label, cost):
  '''
  Get logreg cost of softmax layer
  @param output:Output of softmax layer, or the entire network
  @param label:Label of images
  @param cose:destination array

  Since softmax is applying replica parallel, the output should be global and group shared
  '''
  assert output.global_unique == False and output.group_unique == False
  garray.logreg_cost_col(output.DATA, label, cost.DATA)

def convert_from_data(input, output):
  '''
  Convert input of data layer to output of data layer
  @param input:VArray, segmented in batch dimension
  @param output:VArray, this distribution is decided by next layer
  '''
  _ = time.time()
  assert isinstance(output, VArray)
  assert input.global_unique == False and input.group_unique == True
  if input.check_param(output):
    # when the output is divided in batch dimension
    garray.convert_from_data(input.DATA, output.DATA)
  elif output.global_unique == False and output.group_unique == False:
    # when conv Layer is model parallelism
    input.gather()
    garray.convert_from_data(input.DATA, output.DATA)
  else:
    # input is global shared and group_unique
    if input.global_unique != output.global_unique:
      # There're multi group in output, must regroup the input
      input.regroup_like(output)
    if input.group_slice_dim == output.group_slice_dim:
      garray.convert_from_data(input.DATA, output.DATA)
    else:
      group_output = garray.empty(shape = output.group_area.shape, dtype = np.float32)
      input.group_gather()
      garray.convert_from_data(input.DATA, group_output)
      output.copy_from_group(group_output)

  driver.Context.synchronize()
  if not INNER: MONITOR.add_comm(time.time() - _)

def convert_from_backend(weight, backend):
  '''
  Same as convert_from_backed in garray
  '''
  return garray.convert_from_backend(weight, backend)

def fcforward(input, output, weight, bias, prev_conv):
  _ = time.time()
  input_data = input.local_cache
  garray.matrixmult(weight.DATA, input_data, dest = output.DATA)
  garray.copy_to(output.DATA + bias.DATA , output.DATA)
  driver.Context.synchronize()
  MONITOR.add_fprop(time.time() - _)

def fcbackward(input, weight, grad, out_grad, weight_grad, bias_grad, prev_conv):
  input_data = input.local_cache
  if not out_grad.has_local_cache() :
    if out_grad.DATA.shape != input_data.shape:
      out_grad.local_cache = garray.empty_like(input_data)
    else:
      out_grad.local_cache = out_grad.DATA
  out_grad_data = out_grad.local_cache
  out_grad_data.fill(0)

  # out grad
  _ = time.time()
  garray.matrixmult(garray.transpose(weight.DATA), grad.DATA, dest = out_grad_data)
  driver.Context.synchronize()
  MONITOR.add_bprop(time.time() - _)

  # weight and bias
  _ = time.time()
  garray.matrixmult(grad.DATA, garray.transpose(input.local_cache), dest = weight_grad.DATA)
  garray.copy_to(garray.sum(grad.DATA, axis = 1), bias_grad.DATA)
  driver.Context.synchronize()
  MONITOR.add_wprop(time.time() - _)

def softmax(input, output):
  # softmax is replica parallel, so is previous fc
  local_input = input.DATA
  local_output = output.DATA

  _ = time.time()
  max_rst = garray.max(local_input, axis = 0)
  garray.copy_to(local_input - max_rst, local_output)
  garray.iexp(local_output)
  sum_rst = garray.sum(local_output, axis = 0)
  garray.copy_to(local_output / sum_rst, local_output)
  MONITOR.add_fprop(time.time() - _)

def softmax_bprop(output, label, out_grad):
  # softmax is replica parallel, so is previous fc
  _ = time.time()
  garray.softmax_bprop(output.DATA, label, out_grad.DATA)
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
  _ = time.time()
  input_data = input.local_cache
  garray.convolution(input_data, filter.DATA, output.DATA, bias.DATA, padding, stride)
  MONITOR.add_fprop(time.time() - _)

def bconvolution(input, grad, filter, out_grad, padding, stride):
  input_data = input.local_cache

  if not out_grad.has_local_cache():
    if out_grad.DATA.shape != input_data.shape:
      out_grad.local_cache = garray.empty_like(input_data)
    else:
      out_grad.local_cache = out_grad.DATA
  out_grad_data = out_grad.local_cache

  _ = time.time()
  out_grad_data.fill(0)
  garray.bconvolution(input_data, grad.DATA, filter.DATA, out_grad_data, padding, stride)
  MONITOR.add_bprop(time.time() - _)

def wconvolution(input, grad, weight_grad, bias_grad, padding, stride, *args):
  _ = time.time()
  input_data = input.local_cache

  weight_grad.fill(0)
  bias_grad.fill(0)

  garray.wconvolution(input_data, grad.DATA, weight_grad.DATA, bias_grad.DATA, padding, stride, *args)
  MONITOR.add_wprop(time.time() - _)

def maxpool(input, output, pool_size, start, stride):
  input_data = input.local_cache
  _ = time.time()
  garray.maxpool(input_data, output.DATA, pool_size, start, stride)
  MONITOR.add_fprop(time.time() - _)

def maxundo(input, grad, output, out_grad, pool_size, start, stride):
  input_data = input.local_cache

  if not out_grad.has_local_cache():
    if out_grad.DATA.shape != input_data.shape:
      out_grad.local_cache = garray.empty_like(input_data)
    else:
      out_grad.local_cache = out_grad.DATA
  out_grad_data = out_grad.local_cache

  _ = time.time()
  out_grad_data.fill(0)
  garray.maxundo(input_data, grad.DATA, output.DATA, out_grad_data, pool_size, start, stride)

  MONITOR.add_bprop(time.time() - _)

def avgpool(input, output, pool_size, start, stride):
  input_data = input.local_cache
  _ = time.time()
  garray.avgpool(input_data, output.DATA, pool_size, start, stride)
  MONITOR.add_fprop(time.time() - _)

def avgundo(input, grad, out_grad, pool_size, start, stride):
  input_data = input.local_cache

  if not out_grad.has_local_cache():
    if out_grad.DATA.shape != input_data.shape:
      out_grad.local_cache = garray.empty_like(input_data)
    else:
      out_grad.local_cache = out_grad.DATA
  out_grad_data = out_grad.local_cache

  _ = time.time()
  out_grad_data.fill(0)
  garray.avgundo( input_data, grad.DATA, out_grad_data, pool_size, start, stride)
  MONITOR.add_bprop(time.time() - _)

def rnorm(input, denom, output, size, scalar, pow):
  input_data = input.local_cache

  _ = time.time()
  garray.rnorm(input_data, denom.DATA, output.DATA, size, scalar, pow)
  MONITOR.add_fprop(time.time() - _)

def rnormundo(grad, denom, input, output, out_grad, size, scalar, pow):
  input_data = input.local_cache

  if not out_grad.has_local_cache():
    if out_grad.DATA.shape != input_data.shape:
      out_grad.local_cache = garray.empty_like(input_data)
    else:
      out_grad.local_cache = out_grad.DATA
  out_grad_data = out_grad.local_cache

  _ = time.time()
  out_grad_data.fill(0)
  garray.rnormundo(grad.DATA, denom.DATA, input_data, output.DATA, out_grad_data, size,  scalar, pow)
  MONITOR.add_bprop(time.time() - _)

def rnormcrossmap(input, denom, output, size, scalar, pow, blocked):
  input_data = input.local_cache
  _ = time.time()
  garray.rnormcrossmap(input_data, denom.DATA, output.DATA, size,  scalar, pow, blocked)
  MONITOR.add_fprop(time.time() - _)

def rnormcrossmapundo(grad, denom, input, output, out_grad, size,  scalar, pow, blocked):
  input_data = input.local_cache

  if not out_grad.has_local_cache():
    if out_grad.DATA.shape != input_data.shape:
      out_grad.local_cache = garray.empty_like(input_data)
    else:
      out_grad.local_cache = out_grad.DATA
  out_grad_data = out_grad.local_cache

  _ = time.time()
  out_grad_data.fill(0)

  garray.rnormcrossmapundo(grad.DATA, denom.DATA, input_data, output.DATA, out_grad_data, size,  scalar, pow, blocked)
  MONITOR.add_bprop(time.time() - _)
