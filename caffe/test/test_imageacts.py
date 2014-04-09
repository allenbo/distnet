import caffe
from pycuda import gpuarray, driver
import numpy as np
from distbase.util import divup
caffe.init()

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
  outgrad_shape = (batch_size, color, image_size, image_size)
  filter_shape = (channel, color, filter_size, filter_size)
  output_size = 1 + divup(2 * padding + image_size - filter_size, stride)
  ingrad_shape = (batch_size, channel, output_size, output_size)

  outgrad_local = np.zeros(outgrad_shape).astype(np.float32)
  filter_local = np.ones(filter_shape).astype(np.float32)
  ingrad_local = np.ones(ingrad_shape).astype(np.float32)

  outgrad = gpuarray.to_gpu(outgrad_local)
  filter = gpuarray.to_gpu(filter_local)
  ingrad = gpuarray.to_gpu(ingrad_local)

  print 'ingrad.shape', ingrad.shape
  print 'outgrad.shape', outgrad.shape

  caffe.convImgActs(ingrad, filter, outgrad, image_size, image_size, output_size, -padding, stride,
      color, 1)
  driver.Context.synchronize()
  print 'ImgActs pass the test'
