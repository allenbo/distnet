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
  input_shape = (color, image_size, image_size, batch_size)

  start = 0
  pool_size = 3
  stride = 2

  output_size = divup(image_size - pool_size - start, stride) + 1
  output_shape = (color, output_size, output_size, batch_size)

  input_local = np.ndarray(input_shape).astype(np.float32)
  outgrad_local = np.ndarray(input_shape).astype(np.float32)
  output_local = np.ndarray(output_shape).astype(np.float32)
  ingrad_local = np.ndarray(output_shape).astype(np.float32)

  input = gpuarray.to_gpu(input_local)
  outgrad = gpuarray.to_gpu(outgrad_local)
  output = gpuarray.to_gpu(output_local)
  ingrad = gpuarray.to_gpu(ingrad_local)

  print 'input.shape', input.shape
  print 'output.shape', output.shape

  cudaconv.convLocalMaxUndo(input, ingrad, output, outgrad, pool_size, start, stride, output_size,
      output_size, image_size)
  driver.Context.synchronize()
  print 'max undo pass the test'

  cudaconv.convLocalAvgUndo(ingrad, outgrad, pool_size, start, stride, output_size,
      output_size, image_size, image_size)
  driver.Context.synchronize()
  print 'max undo pass the test'
