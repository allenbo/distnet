from pycuda import gpuarray, driver
import numpy as np
from distbase.util import divup
import caffe
caffe.init()

BATCH = 128 
batch_size = 128

image_sizes = [13, 13]
colors = [96, 256]
for image_size, color in zip(image_sizes, colors):
  size = 5
  scale = 0.0001
  pow = 0.75
  scalar = scale/(size * size)

  input_shape = (batch_size, color, image_size, image_size)
  output_size  = image_size
  output_shape = input_shape

  print 'build input/output/denom'
  input_local = np.random.randint(10, size = input_shape).astype(np.float32)
  denom_local = np.zeros(output_shape).astype(np.float32)
  output_local = np.zeros(output_shape).astype(np.float32)

  input = gpuarray.to_gpu(input_local)
  output = gpuarray.to_gpu(output_local)
  denom = gpuarray.to_gpu(denom_local)

  print 'input.shape', input.shape
  print 'finished'

  print 'gpu computation for RNorm'
  caffe.convResponseNorm(input, denom, output, color, size, image_size, scalar, pow)
  driver.Context.synchronize()
  print 'finished'

  print 'cpu computation for RNorm'
  batch_size = 1
  for b in range(batch_size):
    for c in range(color):
      for x in range(output_size):
        for y in range(output_size):
          start_x = max(x - size / 2, 0)
          start_y = max(y - size / 2, 0)
          end_x = min(x - size / 2 + size, input.shape[1])
          end_y = min(y - size / 2 + size, input.shape[2])
          o = 1 + scalar * (input_local[b, c, start_x:end_x, start_y:end_y] ** 2).sum()
          denom_local[b, c, x, y] = o ** pow
          output_local[b, c, x, y] = input_local[b, c, x, y] / (o ** pow)

  print 'finished'
  diff = output.get()[0, :, :, :] - output_local[0, :, :, :]
  assert(diff < 1e5).all()
  diff = denom.get()[0, :, :, :] - denom_local[0, :, :, :]
  assert(diff < 1e5).all()
  print 'Response Norm passed the test'

  print 'reset output/denom to 0'
  output.fill(0.0)
  output_local.fill(0.0)
  denom.fill(0.0)
  denom_local.fill(0.0)

  print 'gpu computation for cross map'
  scalar = scale / size
  caffe.convResponseNormCrossMap(input, denom, output, color, size, image_size, scalar, pow, False)
  driver.Context.synchronize()
  print 'finished'

  print 'cpu computation for cross map'
  batch_size = 1
  for b in range(batch_size):
    for c in range(color):
      for x in range(output_size):
        for y in range(output_size):
          start_c = max(c - size / 2, 0)
          end_c = min(c - size / 2 + size, input.shape[0])
          o = 2 + scalar * (input_local[b, start_c:end_c, x, y] ** 2).sum()
          output_local[b, c, x, y] = input_local[b, c, x, y] / (o ** pow)

  print 'finished'
  diff = output.get()[0, :, :, :] - output_local[0, :, :, :]
  assert(diff < 1e5).all()
  diff = denom.get()[0, :, :, :] - denom_local[0, :, :, :]
  assert(diff < 1e5).all()
  print 'Response Norm Cross Map passed the test'

  batch_size = BATCH
