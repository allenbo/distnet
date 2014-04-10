from pycuda import gpuarray, driver
import numpy as np
import cudaconv
from distbase.util import divup
cudaconv.init()

BATCH = 128
batch_size = BATCH

colors = [128, 96, 256]
image_sizes = [110, 55, 13]
for image_size, color in zip(image_sizes, colors):
  start = 0
  pool_size = 3
  stride = 2
  input_shape = (color, image_size, image_size, batch_size)

  output_size = divup(image_size - pool_size - start, stride) + 1
  output_shape = (color, output_size, output_size, batch_size)

  input_local = np.random.randn(*input_shape).astype(np.float32)
  output_local = np.random.randn(*output_shape).astype(np.float32)

  input = gpuarray.to_gpu(input_local)
  output = gpuarray.to_gpu(output_local)

  print 'input.shape', input.shape
  print 'output.shape', output.shape

  cudaconv.convLocalMaxPool(input, output, color, pool_size, start, stride, image_size, output_size,
      output_size)
  driver.Context.synchronize()


  batch_size = 1
  for b in range(batch_size):
    for c in range(color):
      for x in range(output_size):
        for y in range(output_size):
          start_x = stride * x
          start_y = stride * y
          end_x = start_x + pool_size if start_x + pool_size < input.shape[1] else input.shape[1]
          end_y = start_y + pool_size if start_y + pool_size < input.shape[2] else input.shape[2]
          output_local[c, x, y, b] = input_local[c, start_x:end_x, start_y:end_y, b].max()

  diff = output.get()[:, :, :, 0] - output_local[:, :, :, 0]
  diff = diff / np.abs(output_local[:, :, :, 0])
  assert (diff < 1e3).all()
  print 'Maxpooling passed the test'


  cudaconv.convLocalAvgPool(input, output, color, pool_size, start, stride, image_size, output_size,
      output_size)
  driver.Context.synchronize()

  batch_size = 1
  for b in range(batch_size):
    for c in range(color):
      for x in range(output_size):
        for y in range(output_size):
          start_x = stride * x
          start_y = stride * y
          end_x = start_x + pool_size if start_x + pool_size < input.shape[1] else input.shape[1]
          end_y = start_y + pool_size if start_y + pool_size < input.shape[2] else input.shape[2]
          output_local[c, x, y, b] = input_local[c, start_x:end_x, start_y:end_y, b].mean()

  diff = output.get()[:, :, :, 0] - output_local[:, :, :, 0]
  diff = diff / np.abs(output_local[:, :, :, 0])
  assert (diff < 1e3).all()
  print 'Avgpooling passed the test'
  batch_size = BATCH
