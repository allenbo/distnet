from .caffe import *
from distbase import cuda_base
from distbase.util import reshape_first, reshape_last
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

backend_name = 'caffe'


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

def convFilterActs(input, weight, output, bias, padding, stride):
  image_y = input.shape[ConvDataLayout.HEIGHT]
  output_y = output.shape[ConvDataLayout.HEIGHT]
  output_x = output.shape[ConvDataLayout.WIDTH]
  color = input.shape[ConvDataLayout.CHANNEL]
  group = 1
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


def convImgActs(input, ingrad, weight, outgrad, padding, stride):
  image_y =  input.shape[ConvDataLayout.HEIGHT]
  image_x =  input.shape[ConvDataLayout.WIDTH]
  output_y =  ingrad.shape[ConvDataLayout.HEIGHT]
  color = input.shape[ConvDataLayout.CHANNEL]
  group = 1

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

def convWeightActs(input, ingrad, weight_grad, bias_grad, padding, stride, color, *args):
  image_y = input.shape[ConvDataLayout.HEIGHT]
  output_y =  ingrad.shape[ConvDataLayout.HEIGHT]
  output_x =  ingrad.shape[ConvDataLayout.WIDTH]
  filter_size =  weight_grad.shape[FilterLayout.HEIGHT]
  color = input.shape[ConvDataLayout.CHANNEL]
  group = 1

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

def convLocalMaxPool(input, output, size, start, stride):
  color =  input.shape[ConvDataLayout.CHANNEL]
  image_y =  input.shape[ConvDataLayout.HEIGHT]
  output_y = output.shape[ConvDataLayout.HEIGHT]
  output_x =  output.shape[ConvDataLayout.WIDTH]

  caffe.convLocalMaxPool(input, output, color, size, start, stride, image_y, output_y, output_x)


def convLocalMaxUndo(input, grad, output, outgrad, size, start, stride):
  output_y = output.shape[ConvDataLayout.HEIGHT]
  output_x = output.shape[ConvDataLayout.WIDTH]
  image_y = input.shape[ConvDataLayout.HEIGHT]

  caffe.convLocalMaxUndo(input, grad, output, outgrad, size, start, stride, output_y, output_x, image_y)

def convLocalAvgPool(input, output, size, start, stride):
  color = input.shape[ConvDataLayout.CHANNEL]
  image_y = input.shape[ConvDataLayout.HEIGHT]
  output_y = output.shape[ConvDataLayout.HEIGHT]
  output_x = output.shape[ConvDataLayout.WIDTH]

  caffe.convLocalAvgPool(input, output, color, size, start, stride, image_y, output_y, output_x)

def convLocalAvgUndo(input, grad, outgrad, size, start, stride):
  output_y = grad.shape[ConvDataLayout.HEIGHT]
  output_x = grad.shape[ConvDataLayout.WIDTH]
  image_y = input.shape[ConvDataLayout.HEIGHT]
  image_x = input.shape[ConvDataLayout.WIDTH]

  caffe.convLocalAvgUndo(grad, outgrad, size, start, stride, output_y, output_x, image_y, image_x)

def convResponseNorm(input, denom, output, size, scalar, pow):
  color = input.shape[ConvDataLayout.CHANNEL]
  image_y = input.shape[ConvDataLayout.HEIGHT]
  caffe.convResponseNorm(input, denom, output, color, size, image_y, scalar, pow)

def convResponseNormUndo(grad, denom, input, output, outgrad, size, scalar, pow):
  color = input.shape[ConvDataLayout.CHANNEL]
  image_y = input.shape[ConvDataLayout.HEIGHT]
  caffe.convResponseNormUndo(grad, denom, input, output, outgrad, color, size, image_y, scalar, pow, 0.0, 1.0)

def convResponseNormCrossMap(input, denom, output, size, scalar, pow, blocked):
  color = input.shape[ConvDataLayout.CHANNEL]
  image_y = input.shape[ConvDataLayout.HEIGHT]
  caffe.convResponseNormCrossMap(input, denom, output,color, size, image_y, scalar, pow, blocked)

def convResponseNormCrossMapUndo(grad, denom, input, output, outgrad, size, scalar, pow, blocked):
  color = input.shape[ConvDataLayout.CHANNEL]
  image_y = input.shape[ConvDataLayout.HEIGHT]
  caffe.convResponseNormCrossMapUndo(grad, denom, input, output, outgrad, color, size, image_y, scalar, pow, blocked, 0.0, 1.0)

def convert_to_fc(input):
  batch_size = input.shape[ConvDataLayout.BATCH]
  new_shape = (batch_size, int(np.prod(input.shape) / batch_size))
  rst = cuda_base.transpose(input.reshape(new_shape))
  return rst

def convert_to_conv(input):
  shape = reshape_first(input).shape
  real_shape = tuple(shape[::-1])
  real_input = input.reshape(real_shape)
  return cuda_base.transpose(real_input)

def convert_from_data(input, output):
  cuda_base.transpose(reshape_last(input), dst = reshape_first(output))

def convert_from_backend(weight, backend):
  if backend == backend_name:
    return weight
  if backend == 'cudaconv':
    old_shape = list(weight.shape)
    new_shape = old_shape[-1:] + old_shape[:-1]
    new_weight = reshape_last(weight)
    new_weight = new_weight.transpose().copy()
    return new_weight.reshape(tuple(new_shape))

  assert False, 'Not implemented'
