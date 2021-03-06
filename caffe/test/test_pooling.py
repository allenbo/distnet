import caffe
from pycuda import gpuarray, driver
import numpy as np
from distbase.util import divup
caffe.init()

BATCH = 128
batch_size = BATCH

colors = [128, 96, 256]
image_sizes = [110, 55, 13]

for image_size, color in zip(image_sizes, colors):
  start = 0
  pool_size = 3
  stride = 2

  input_shape = (batch_size, color, image_size, image_size)
  output_size = divup(image_size - pool_size - start, stride) + 1
  output_shape = (batch_size, color, output_size, output_size)

  print 'build input/output'
  input_local = np.random.randint(10, size = input_shape).astype(np.float32)
  output_local = np.random.randint(10, size = output_shape).astype(np.float32)

  input = gpuarray.to_gpu(input_local)
  output = gpuarray.to_gpu(output_local)
  print 'input.shape', input.shape
  print 'output.shape', output.shape
  print 'finished'

  print 'gpu computation for maxpool'
  caffe.convLocalMaxPool(input, output, color, pool_size, start, stride, image_size, output_size,
      output_size)
  print 'finished'

  print 'cpu computation for maxpool'

  batch_size = 1
  for b in range(batch_size):
    for c in range(color):
      for x in range(output_size):
        for y in range(output_size):
          start_x = stride * x
          start_y = stride * y
          end_x = start_x + pool_size if start_x + pool_size < input.shape[2] else input.shape[2]
          end_y = start_y + pool_size if start_y + pool_size < input.shape[3] else input.shape[3]
          output_local[b, c, x, y] = input_local[b, c, start_x:end_x, start_y:end_y].max()
  print 'finished'

  diff = output.get()[0, :, :, :] - output_local[0, :, :, :]
  diff = diff / np.abs(output_local[0, :, :, :])
  assert (diff < 1e3).all()
  print 'Maxpooling passed the test'
  
  print 'reset output to 0'
  output.fill(0.0)
  output_local.fill(0.0)

  print 'gpu computation for avgpool'
  caffe.convLocalAvgPool(input, output, color, pool_size, start, stride, image_size, output_size,
      output_size)
  print 'finished'

  print 'cpu computation for avgpool'

  batch_size = 1
  for b in range(batch_size):
    for c in range(color):
      for x in range(output_size):
        for y in range(output_size):
          start_x = stride * x
          start_y = stride * y
          end_x = start_x + pool_size if start_x + pool_size < input.shape[2] else input.shape[2]
          end_y = start_y + pool_size if start_y + pool_size < input.shape[3] else input.shape[3]
          output_local[b, c, x, y] = input_local[b, c, start_x:end_x, start_y:end_y].mean()

  print 'finished'
  diff = output.get()[0, :, :, :] - output_local[0, :, :, :]
  diff = diff / np.abs(output_local[0, :, :, :])
  assert (diff < 1e3).all()
  print 'Avgpooling passed the test'
  batch_size = BATCH
