from pycuda import gpuarray, driver, autoinit
import numpy as np
import cudaconv
from distnet.util import divup


batch_size = 128
image_size = 27
color = 96
input_shape = (color, image_size, image_size, batch_size)

size = 5
scale = 0.0001
pow = 0.75
scaler = scale/(size * size)

output_size  = image_size
output_shape = input_shape

input_local = np.random.randn(*input_shape).astype(np.float32)
output_local = np.random.randn(*output_shape).astype(np.float32)
denom_local = np.random.randn(*output_shape).astype(np.float32)

input = gpuarray.to_gpu(input_local)
output = gpuarray.to_gpu(output_local)
denom = gpuarray.to_gpu(denom_local)

cudaconv.convResponseNorm(input, denom, output, color, size, image_size, scaler, pow)

batch_size = 1
for b in range(batch_size):
  for c in range(color):
    for x in range(output_size):
      for y in range(output_size):
        start_x = x - size / 2
        start_y = y - size / 2
        end_x = min(start_x + size, input.shape[1])
        end_y = min(start_y + size, input.shape[2])
        start_x = max(0, start_x)
        start_y = max(0, start_y)
        tmp = input_local[c: start_x:end_x, start_y:end_y, b]
        o = (tmp ** 2).sum()
        o = (2 + scaler * o)
        output_local[c, x, y, b] = input_local[c, x, y, b] / o

diff = output.get()[:, :, :, 0] = output_local[:, :, :, 0]
diff = diff / np.abs(output_local[:, :, :, 0])
assert(diff < 1e5).all()
print 'Response Norm passed the test'
