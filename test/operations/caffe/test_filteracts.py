import caffe
import cudaconv
from pycuda import gpuarray, driver, autoinit
import numpy as np
from distnet.util import divup
np.set_printoptions(threshold = np.nan)

batch_size = 128
image_size = 224
color = 3
input_shape = (batch_size, color, image_size, image_size)

filter_size  = 11
channel = 96
filter_shape = (channel, color, filter_size, filter_size)

padding = 0
stride = 4
output_size = 1 + divup(2 * padding + image_size - filter_size, stride)
output_shape = (batch_size, channel, output_size, output_size)

input_local = np.random.randn(*input_shape).astype(np.float32)
filter_local = np.random.randn(*filter_shape).astype(np.float32)
output_local = np.zeros(output_shape).astype(np.float32)

input_caffe = gpuarray.to_gpu(input_local)
filter_caffe = gpuarray.to_gpu(filter_local)
output_caffe = gpuarray.to_gpu(output_local)

caffe.convFilterActs(input_caffe, filter_caffe, output_caffe, image_size, output_size, output_size, -padding,
    stride, color, 1)

if padding != 0:
  tmp_shape  = (batch_size, channel, image_size + 2 * padding, image_size + 2 * padding)
  tmp_input = np.zeros(tmp_shape).astype(np.float32)
  tmp_input[:, :, padding:padding+image_size, padding:padding+image_size] = input_local
  input_local = tmp_input


batch_size = 1
for b in range(batch_size):
  for c in range(channel):
    for x in range(output_size):
      for y in range(output_size):
        start_x = stride * x
        start_y = stride * y
        o = 0
        f = filter_local
        if start_x + filter_size >= input_local.shape[2]:
          f = f[:, :, :input_local.shape[2]-start_x, :]
        if start_y + filter_size >= input_local.shape[3]:
          f = f[:, :, :, :input_local.shape[3]-start_y]
        for cr in range(color):
          left = input_local[b, cr, start_x:start_x+filter_size, start_y:start_y+filter_size]
          right = f[c, cr, :, :]
          o += (left * right).sum()
        output_local[b, c, x, y] = o

diff = output_caffe.get()[0, :, :, :] - output_local[0, :, :, :]
assert (diff < 1e-3).all()
print 'Convolution passed the test'
