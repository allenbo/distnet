from .cudaconv3 import *
import cudaconv3
from distbase.util import reshape_first, reshape_last, divup
import numpy as np


class ConvDataLayout(object):
  CHANNEL = 0
  HEIGHT = 1
  WIDTH = 2
  BATCH = 3
  DIM = 4

  @staticmethod
  def get_output_shape(image_y, image_x, channel, batch_size):
    return (channel, image_y, image_x, batch_size)


class FilterLayout(object):
  CHANNEL = 0
  HEIGHT = 1
  WIDTH = 2
  NUM = 3
  DIM = 4

  @staticmethod
  def get_filter_shape(filter_size, channel, num):
    return (channel, filter_size, filter_size, num)

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

backend_name = 'cudaconv3'

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
  from distbase import cuda_base
  image_y = input.shape[ConvDataLayout.HEIGHT]
  output_y = output.shape[ConvDataLayout.HEIGHT]
  output_x = output.shape[ConvDataLayout.WIDTH]
  color = input.shape[ConvDataLayout.CHANNEL]

  cudaconv3.convFilterActs(input, weight, output, image_y, output_y, output_x, padding, stride, color, 1)
  batch_size = output.shape[ConvDataLayout.BATCH]
  channel = output.shape[ConvDataLayout.CHANNEL]

  # bias term
  cuda_base.add_vec_to_rows(output.reshape((channel, output_y * output_x * batch_size)), bias)

def convImgActs(input, grad, weight, out_grad, padding, stride):
  image_y =  input.shape[ConvDataLayout.HEIGHT]
  image_x =  input.shape[ConvDataLayout.WIDTH]
  output_y =  grad.shape[ConvDataLayout.HEIGHT]
  color = input.shape[ConvDataLayout.CHANNEL]

  cudaconv3.convImgActs(grad, weight, out_grad, image_y, image_x, output_y, padding, stride, color, 1)

def convWeightActs(input, ingrad, weight_grad, bias_grad, padding, stride, sum_width):
  image_y = input.shape[ConvDataLayout.HEIGHT]
  output_y =  ingrad.shape[ConvDataLayout.HEIGHT]
  output_x =  ingrad.shape[ConvDataLayout.WIDTH]
  filter_size =  weight_grad.shape[FilterLayout.HEIGHT]
  color = input.shape[ConvDataLayout.CHANNEL]
  group = 1

  batch_size = ingrad.shape[ConvDataLayout.BATCH]
  channel = ingrad.shape[ConvDataLayout.CHANNEL]

  do_partial_sum =  sum_width < output_x
  target = weight_grad
  if do_partial_sum:
    output_chunk = divup(output_x, sum_width) * divup(output_y, sum_width)
    shape = (output_chunk *  filter_size * filter_size * color / group, channel)
    from pycuda import gpuarray
    target = gpuarray.GPUArray(shape = shape, dtype = np.float32)

  cudaconv3.convWeightActs(input, ingrad, target, image_y, output_y, output_x, filter_size, padding, stride, color, group, sum_width)
  if do_partial_sum:
    target = target.reshape((output_chunk, filter_size * filter_size * color / group * channel))
    weight_shape = weight_grad.shape
    weight_grad = weight_grad.reshape((1, filter_size * filter_size * color * channel))
    from distbase import cuda_base
    cuda_base.add_col_sum_to_vec(weight_grad, target, alpha = 0, beta = 1.0)

  cudaconv3.sum(ingrad.reshape((channel, output_y * output_x * batch_size)), 1, bias_grad)

def convLocalMaxPool(input, output, size, start, stride):
  color =  input.shape[ConvDataLayout.CHANNEL]
  image_y =  input.shape[ConvDataLayout.HEIGHT]
  output_y = output.shape[ConvDataLayout.HEIGHT]
  output_x =  output.shape[ConvDataLayout.WIDTH]
  cudaconv3.convLocalMaxPool(input, output, color, size, start, stride, image_y, output_y, output_x)


def convLocalMaxUndo(input, grad, output, outgrad, size, start, stride):
  output_y = output.shape[ConvDataLayout.HEIGHT]
  output_x = output.shape[ConvDataLayout.WIDTH]
  image_y = input.shape[ConvDataLayout.HEIGHT]

  cudaconv3.convLocalMaxUndo(input, grad, output, outgrad, size, start, stride, output_y, output_x, image_y)

def convLocalAvgPool(input, output, size, start, stride):
  color = input.shape[ConvDataLayout.CHANNEL]
  image_y = input.shape[ConvDataLayout.HEIGHT]
  output_y = output.shape[ConvDataLayout.HEIGHT]
  output_x = output.shape[ConvDataLayout.WIDTH]

  cudaconv3.convLocalAvgPool(input, output, color, size, start, stride, image_y, output_y, output_x)

def convLocalAvgUndo(input, grad, outgrad, size, start, stride):
  output_y = grad.shape[ConvDataLayout.HEIGHT]
  output_x = grad.shape[ConvDataLayout.WIDTH]
  image_y = input.shape[ConvDataLayout.HEIGHT]
  image_x = input.shape[ConvDataLayout.WIDTH]
  cudaconv3.convLocalAvgUndo(grad, outgrad, size, start, stride, output_y, output_x, image_y, image_x)

def convResponseNorm(input, denom, output, size, scalar, pow):
  color = input.shape[ConvDataLayout.CHANNEL]
  image_y = input.shape[ConvDataLayout.HEIGHT]
  cudaconv3.convResponseNorm(input, denom, output, color, size, image_y, scalar, pow)

def convResponseNormUndo(grad, denom, input, output, outgrad, size, scalar, pow):
  color = input.shape[ConvDataLayout.CHANNEL]
  image_y = input.shape[ConvDataLayout.HEIGHT]
  cudaconv3.convResponseNormUndo(grad, denom, input, output, outgrad, color, size, image_y, scalar, pow, 0.0, 1.0)

def convResponseNormCrossMap(input, denom, output, size, scalar, pow, blocked):
  color = input.shape[ConvDataLayout.CHANNEL]
  image_y = input.shape[ConvDataLayout.HEIGHT]
  cudaconv3.convResponseNormCrossMap(input, denom, output,color, size, image_y, scalar, pow, blocked)

def convResponseNormCrossMapUndo(grad, denom, input, output, outgrad, size, scalar, pow, blocked):
  color = input.shape[ConvDataLayout.CHANNEL]
  image_y = input.shape[ConvDataLayout.HEIGHT]
  cudaconv3.convResponseNormCrossMapUndo(grad, denom, input, output, outgrad, color, size, image_y, scalar, pow, blocked, 0.0, 1.0)

def convert_to_fc(input):
  return input

def convert_to_conv(input):
  return input

def convert_from_data(input, output):
  from distbase import cuda_base
  cuda_base.gpu_copy_to(input, output)

def convert_from_backend(weight, backend):
  if backend != 'caffe':
    return weight
  from distbase import cuda_base
  old_shape = list(weight.shape)
  new_shape = old_shape[1:] + old_shape[:1]
  new_weight = reshape_first(weight).transpose().copy()
  return new_weight.reshape(tuple(new_shape))
