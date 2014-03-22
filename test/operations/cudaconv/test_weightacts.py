import cudaconv
from pycuda import gpuarray, driver, autoinit
import numpy as np
from distnet.util import divup
np.set_printoptions(threshold = np.nan)

batch_size = 128
image_size = 224
color = 3
input_shape = (color, image_size, image_size, batch_size)

filter_size  = 11
channel = 96
filter_shape = (color, filter_size, filter_size, channel)

padding = 1
stride = 4
output_size = 1 + divup(2 * padding + image_size - filter_size, stride)
ingrad_shape = (channel, output_size, output_size, batch_size)

input_local = np.ones(input_shape).astype(np.float32)
ingrad_local = np.ones(ingrad_shape).astype(np.float32)
filter_local = np.zeros(filter_shape).astype(np.float32)

for c in range(channel):
  for b in range(batch_size):
    ingrad_local[c, :, :, b] *= np.arange(output_size).reshape((output_size, 1))

input_cudaconv = gpuarray.to_gpu(input_local)
ingrad_cudaconv = gpuarray.to_gpu(ingrad_local)
filter_cudaconv = gpuarray.to_gpu(filter_local)

cudaconv.convWeightActs(input_cudaconv, ingrad_cudaconv, filter_cudaconv, image_size, output_size,
    output_size, filter_size, -padding, stride, color, 1, 0)

print filter_cudaconv.get()[0, :, :, 0]
