from pycuda import gpuarray, driver, autoinit
import numpy as np
import time
import garray

batch_size = 128
for batch_size in [32, 64, 128, 256]:
  image_size = 55
  color = 3
  input_shape = (color * image_size * image_size, batch_size)
  output_shape = input_shape

  e = 0.0
  input_local = np.random.randn(*input_shape).astype(np.float32)
  output_local = np.random.randn(*output_shape).astype(np.float32)

  input = gpuarray.to_gpu(input_local)
  output = gpuarray.to_gpu(output_local)
  print 'batch_size = %d' % batch_size
  count = 10
  for i in range(count):
    garray.relu_activate(input, output, e)
    driver.Context.synchronize()

  count = 1000
  start = time.time()
  for i in range(count):
    garray.relu_activate(input, output, e)
    driver.Context.synchronize()

  print '%f seconds for %d times relu activate' % (time.time() - start, count)

  data_amount = np.prod(input_shape) * 2 * 4
  print 'load [%fMB] data' % (data_amount * 1.0 / (1 << 20))
