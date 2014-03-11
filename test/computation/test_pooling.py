from pycuda import gpuarray, driver, autoinit
import numpy as np
import cudaconv
from distnet.util import divup
import time
import random

batch_size = 128
batchs = [128]
for batch_size in batchs:
  image_size = 55
  color = 96
  input_shape = (color, image_size, image_size, batch_size)

  start = 0
  pool_size = 3
  stride = 2

  output_size = divup(image_size - pool_size - start, stride) + 1
  output_shape = (color, output_size, output_size, batch_size)

  input_local = np.random.randn(*input_shape).astype(np.float32)
  output_local = np.random.randn(*output_shape).astype(np.float32)

  input = gpuarray.to_gpu(input_local)
  output = gpuarray.to_gpu(output_local)

  count = 10
  for i in range(count):
    cudaconv.convLocalMaxPool(input, output, color, pool_size, start, stride, image_size, output_size,
        output_size)
    driver.Context.synchronize()


  print 'batch_size == %d' % batch_size
  count = 100
  s = time.time()
  for i in range(count):
    cudaconv.convLocalMaxPool(input, output, color, pool_size, start, stride, image_size, output_size,
        output_size)
    driver.Context.synchronize()
  
  print '%f second per convLocalMaxPool' % ((time.time() - s) / count )
  data_amount = np.prod(output_shape) * (pool_size * pool_size + 1) * 4
  comput_amount = np.prod(output_shape) * (pool_size * pool_size + 1)
  print 'load [%fMB] data' % (data_amount * 1.0 / (1 << 20))
  print 'do [%fG] float multiplication' % (comput_amount * 1.0 / 1e9)
