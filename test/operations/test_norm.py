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
        start_x = max(x - size / 2, 0)
        start_y = max(y - size / 2, 0)
        end_x = min(x - size / 2 + size, input.shape[1])
        end_y = min(y - size / 2 + size, input.shape[2])
        o = 2 + scaler * (input_local[c: start_x:end_x, start_y:end_y, b] ** 2).sum()
        denom_local[c, x, y, b] = o ** pow
        output_local[c, x, y, b] = input_local[c, x, y, b] / (o ** pow)

diff = output.get()[:, :, :, 0] = output_local[:, :, :, 0]
diff = diff / np.abs(output_local[:, :, :, 0])
assert(diff < 1e5).all()
diff = denom.get()[:, :, :, 0] = denom_local[:, :, :, 0]
diff = diff / np.abs(denom_local[:, :, :, 0])
assert(diff < 1e5).all()
print 'Response Norm passed the test'

scaler = scale / size
cudaconv.convResponseNormCrossMap(input, denom, output, color, size, image_size, scaler, pow, False)

batch_size = 1
for b in range(batch_size):
  for c in range(color):
    for x in range(output_size):
      for y in range(output_size):
        start_c = max(c - size / 2, 0)
        end_c = min(c - size / 2 + size, input.shape[0])
        o = 2 + scaler * (input_local[start_c:end_c, x, y, b] ** 2).sum()
        output_local[c, x, y, b] = input_local[c, x, y, b] / (o ** pow)

diff = output.get()[:, :, :, 0] = output_local[:, :, :, 0]
diff = diff / np.abs(output_local[:, :, :, 0])
assert(diff < 1e5).all()
diff = denom.get()[:, :, :, 0] = denom_local[:, :, :, 0]
diff = diff / np.abs(denom_local[:, :, :, 0])
assert(diff < 1e5).all()
print 'Response Norm Cross Map passed the test'
