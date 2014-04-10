import cudaconv
from pycuda import gpuarray, driver
import numpy as np
from distbase.util import divup
cudaconv.init()

BATCH = 128
batch_size = BATCH


colors = [3, 96, 128]
channels = [96, 128, 128]
image_sizes = [224, 27, 13]
filter_sizes = [11, 5, 3]
paddings = [0, 2, 1]
strides = [4, 1, 1]

for image_size, color, channel, padding, stride, filter_size in zip(image_sizes, colors, channels, paddings, strides, filter_sizes):
  print 'color = %d channel = %d image_size = %d padding = %d stride = %d' % (color, channel, image_size, padding, stride)

  input_shape = (color, image_size, image_size, batch_size)
  filter_shape = (color, filter_size, filter_size, channel)
  output_size = 1 + divup(2 * padding + image_size - filter_size, stride)
  ingrad_shape = (channel, output_size, output_size, batch_size)

  input_local = np.ones(input_shape).astype(np.float32)
  ingrad_local = np.ones(ingrad_shape).astype(np.float32)
  filter_local = np.zeros(filter_shape).astype(np.float32)

  input_cudaconv = gpuarray.to_gpu(input_local)
  ingrad_cudaconv = gpuarray.to_gpu(ingrad_local)
  filter_cudaconv = gpuarray.to_gpu(filter_local)
  
  print 'input.shape', input_cudaconv.shape
  print 'ingrad.shape', ingrad_cudaconv.shape

  cudaconv.convWeightActs(input_cudaconv, ingrad_cudaconv, filter_cudaconv, image_size, output_size,
      output_size, filter_size, -padding, stride, color, 1, 0)
  driver.Context.synchronize()
