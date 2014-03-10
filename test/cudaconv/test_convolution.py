import cudaconv
from pycuda import gpuarray, driver, autoinit
import numpy as np
from distnet.util import divup
np.set_printoptions(threshold = np.nan)

batch_size = 8
image_size = 224
color = 3
input_shape = (color, image_size, image_size, batch_size)

filter_size  = 11
channel = 96
filter_shape = (color, filter_size, filter_size, channel)

padding = 0
stride = 4
output_size = 1 + divup(2 * padding + image_size - filter_size, stride)
output_shape = (channel, output_size, output_size, batch_size)

input_local = np.ones(input_shape).astype(np.float32)
filter_local = np.ones(filter_shape).astype(np.float32)
output_local = np.zeros(output_shape).astype(np.float32)

input = gpuarray.to_gpu(input_local)
filter = gpuarray.to_gpu(filter_local)
output = gpuarray.to_gpu(output_local)

cudaconv.convFilterActs(input, filter, output, image_size, output_size, output_size, -padding,
    stride, color, 1)

#print output.get()[0, :, :, 0]
#print output.shape
if padding != 0:
  tmp_shape  = (channel, image_size + 2 * padding, image_size + 2 * padding, batch_size)
  tmp_input = np.zeors(tmp_shape).astype(np.float32)
  tmp_input[:, padding:padding+image_size, padding:padding+image_size, :] = input
  input = tmp_input


starts = [stride * i for i in range(output_size)]

batch_size = 1
for b in range(batch_size):
  for c in range(channel):
    for x in range(output_size):
      for y in range(output_size):
        start_x = stride * x
        start_y = stride * y
        o = 0
        f = filter_local
        if start_x + filter_size > input.shape[1]:
          f = f[:, :input.shape[1]-start_x, :, :]
        if start_y + filter_size > input.shape[2]:
          f = f[:, :, :input.shape[2]-start_y, :]
        for cr in range(color):
          left = input_local[cr, start_x:start_x+filter_size, start_y:start_y+filter_size, b]
          right = f[cr, :, :, c]
          o += (left * right).sum()
        output_local[c, x, y, b] = o
  print '%d image finished' % b

#print output_local[0, :, :, 0]
assert (output.get()[:, :, :, 0] - output_local[:, :, :, 0]).sum() < 1e-3
