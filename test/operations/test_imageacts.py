import cudaconv
from pycuda import gpuarray, driver, autoinit
import numpy as np
from distnet.util import divup
np.set_printoptions(threshold = np.nan)

batch_size = 128
image_size = 10
color = 3
outgrad_shape = (color, image_size, image_size, batch_size)

filter_size  = 3
channel = 32
filter_shape = (color, filter_size, filter_size, channel)

padding = 0
stride = 2
output_size = 1 + divup(2 * padding + image_size - filter_size, stride)
ingrad_shape = (channel, output_size, output_size, batch_size)

outgrad_local = np.zeros(outgrad_shape).astype(np.float32)
filter_local = np.ones(filter_shape).astype(np.float32)
ingrad_local = np.ones(ingrad_shape).astype(np.float32)

for cr in range(color):
  for c in range(channel):
    filter_local[cr, :, :, c] = np.arange(filter_size * filter_size).reshape((filter_size,filter_size))


for c in range(channel):
  for b in range(batch_size):
    ingrad_local[c, :, :, b] *= np.arange(output_size).reshape((output_size, 1))

outgrad = gpuarray.to_gpu(outgrad_local)
filter = gpuarray.to_gpu(filter_local)
ingrad = gpuarray.to_gpu(ingrad_local)

cudaconv.convImgActs(ingrad, filter, outgrad, image_size, image_size, output_size, -padding, stride,
    color, 1)

'''
print ingrad.get()[0, :, :, 0]
print filter.get()[0, :, :, 0]
print outgrad.get()[0, :, :, 0]/32
'''

for cr in range(color):
  for c in range(channel):
    tmp = filter_local[cr, :, :, c]
    tmp = tmp.flatten()[::-1]
    filter_local[cr, :, :, c] = tmp.reshape(filter_local[cr, :, :, c].shape)
