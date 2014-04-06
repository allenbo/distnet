import caffe
from pycuda import gpuarray, driver
import numpy as np
from distbase.util import divup
caffe.init()



batch_size = 128
image_size = 10
color = 3
outgrad_shape = (batch_size, color, image_size, image_size)

filter_size  = 3
channel = 32
filter_shape = (channel, color, filter_size, filter_size)

padding = 1
stride = 2
output_size = 1 + divup(2 * padding + image_size - filter_size, stride)
ingrad_shape = (batch_size, channel, output_size, output_size)

outgrad_local = np.zeros(outgrad_shape).astype(np.float32)
filter_local = np.ones(filter_shape).astype(np.float32)
ingrad_local = np.ones(ingrad_shape).astype(np.float32)

for cr in range(color):
  for c in range(channel):
    filter_local[c, cr, :, :] = np.arange(filter_size * filter_size).reshape((filter_size,filter_size))


for c in range(channel):
  for b in range(batch_size):
    ingrad_local[b, c, :, :] *= np.arange(output_size).reshape((output_size, 1))

outgrad = gpuarray.to_gpu(outgrad_local)
filter = gpuarray.to_gpu(filter_local)
ingrad = gpuarray.to_gpu(ingrad_local)

caffe.convImgActs(ingrad, filter, outgrad, image_size, image_size, output_size, -padding, stride,
    color, 1)
