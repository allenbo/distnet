from .caffe import *
from distbase import cuda_base
from pycuda import gpuarray, driver
import numpy as np

class ConvDataLayout(object):
  BATCH = 0
  CHANNEL = 1
  HEIGHT = 2
  WIDTH = 3
  DIM = 4
  
  @staticmethod
  def get_output_shape(image_y, image_x, channel, batch_size):
    return (batch_size, channel, image_y, image_x)
  

class FilterLayout(object):
  NUM = 0
  CHANNEL = 1
  HEIGHT = 2
  WIDTH = 3
  DIM = 4

  @staticmethod
  def get_filter_shape(filter_size, channel, num):
    return (num, channel, filter_size, filter_size)

class FCDataLayout(object):
  NEURON = 0
  BATCH = 1
  DIM = 2

  @staticmethod
  def get_output_shape(neuron, batch_size):
    return (neuron, batch_size)

class WeightLayout(object):
  OUTPUT = 0
  INPUT = 1
  DIM = 2

  @staticmethod
  def get_weight_shape(input, output):
    return (output, input)

backend = 'caffe'


def get_output_shape_4D(image_y, image_x, channel, batch_size):
  return (batch_size, channel, image_y, image_x)

CONTEXT = None

def init(device=-1):
  global CONTEXT
  if CONTEXT is not None:
    return
  
  # MAGIC MAGIC
  from pycuda import driver
  driver.init()
  
  if device == -1:
    from pycuda.tools import make_default_context
    CONTEXT = make_default_context()
    device = CONTEXT.get_device()
  else:
    device = driver.Device(device % driver.Device.count())
    CONTEXT = device.make_context()
  
  #print 'Starting up using device: %s:%s' % (device.name(), device.pci_bus_id()) 
  import atexit
  atexit.register(CONTEXT.detach)
  return CONTEXT

def convFilterActs(input, weight, output, bias, image_y, output_y, output_x, padding, stride, color, group):
  batch_size = input.shape[0]
  image_x = input.shape[-1]
  filter_size = weight.shape[-1]
  num_filter = weight.shape[0] 

  assert group == 1, 'Group has to be 1, now it\'s %d' % group
  col_buffer = gpuarray.zeros(shape = (weight.size / weight.shape[0], output_y * output_x), dtype = np.float32)
  bias_multiplier = gpuarray.GPUArray(shape = (1, output_y * output_x), dtype = np.float32)
  bias_multiplier.fill(1.0)

  for i in range(batch_size):
    im2col_gpu(input.ptr + input.strides[0] * i, color, image_y, image_x, filter_size, stride, padding, col_buffer.ptr)
    output_buffer = gpuarray.GPUArray(shape = (weight.shape[0], output_y * output_x), dtype = np.float32, gpudata = output.ptr + output.strides[0] * i)
    weight_buffer = weight.reshape((num_filter, color * filter_size * filter_size))

    cuda_base.matrixmult(weight_buffer, col_buffer, dest = output_buffer)
    # bias term
    cuda_base.matrixmult(bias, bias_multiplier, dest = output_buffer, alpha = 1.0, beta = 1.0)


def convImgActs(ingrad, weight, outgrad, image_y, image_x, output_y, padding, stride, color,
    group):
  batch_size = ingrad.shape[0]
  output_x = ingrad.shape[-1]
  filter_size = weight.shape[-1]
  num_filter = weight.shape[0]

  assert group == 1, 'Group has to be 1, now it\'s %d' % group
  weight_buffer = cuda_base.transpose(weight.reshape((num_filter, color * filter_size * filter_size)))
  col_buffer = gpuarray.zeros(shape = (color * filter_size * filter_size, output_y * output_x), dtype = outgrad.dtype)

  for i in range(batch_size):
    ingrad_buffer = gpuarray.GPUArray(shape = (num_filter, output_x* output_y), dtype = ingrad.dtype, gpudata = ingrad.ptr + ingrad.strides[0] * i)

    cuda_base.matrixmult(weight_buffer, ingrad_buffer, dest = col_buffer)
    col2im_gpu(col_buffer.ptr, color, image_y, image_x, filter_size, stride, padding, outgrad.ptr + outgrad.strides[0] * i)

def convWeightActs(input, ingrad, weight_grad, bias_grad, image_y, output_y, output_x, filter_size, padding, stride, color, group, partial_sum):
  batch_size = input.shape[0]
  image_x = input.shape[-1]
  filter_size = weight_grad.shape[-1]
  num_filter = weight_grad.shape[0] 

  assert group == 1, 'Group has to be 1, now it\'s %d' % group
  col_buffer = gpuarray.zeros(shape = (weight_grad.size / weight_grad.shape[0], output_y * output_x), dtype = np.float32)
  bias_multiplier = gpuarray.GPUArray(shape = (output_y * output_x, 1), dtype = np.float32)
  bias_multiplier.fill(1.0)
  for i in range(batch_size):
    im2col_gpu(input.ptr + input.strides[0] * i, color, image_y, image_x, filter_size, stride, padding, col_buffer.ptr)

    ingrad_buffer = gpuarray.GPUArray(shape = (num_filter, output_x* output_y), dtype = ingrad.dtype, gpudata = ingrad.ptr + ingrad.strides[0] * i)
    cuda_base.matrixmult(ingrad_buffer, cuda_base.transpose(col_buffer), dest = weight_grad, alpha = 1.0, beta = 1.0)
    # bias term
    cuda_base.matrixmult(ingrad_buffer, bias_multiplier, bias_grad, alpha = 1.0, beta = 1.0)

def convert_to_fc(input):
  batch_size = input.shape[ConvDataLayout.BATCH]
  new_shape = (batch_size, int(np.prod(input.shape) / batch_size))
  rst = cuda_base.transpose(input.reshape(new_shape))
  return rst

def convert_to_conv(input):
  return cuda_base.transpose(input)

def convert_from_data(input, output):
  cuda_base.transpose(input, dst = output) 
