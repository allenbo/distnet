from pycuda import gpuarray, driver
import numpy as np
import time
import garray
garray.device_init()

batch_size = 128
print '%10s\t%10s\t%10s' %('batch', 'data', 'real')
for batch_size in [8, 16, 32, 64, 128, 256]:
  image_size = 55
  color = 3
  input_shape = (color * image_size * image_size, batch_size)
  output_shape = input_shape

  e = 0.0
  input_local = np.random.randn(*input_shape).astype(np.float32)
  output_local = np.random.randn(*output_shape).astype(np.float32)

  input = gpuarray.to_gpu(input_local)
  output = gpuarray.to_gpu(output_local)
  count = 10
  for i in range(count):
    garray.relu_activate(input, output, e)
    driver.Context.synchronize()

  count = 1000
  start = time.time()
  for i in range(count):
    garray.relu_activate(input, output, e)
    driver.Context.synchronize()
  
  real_time = (time.time() - start) / count

  data_amount = (np.prod(input_shape) * 2 * 4.0) / (1<<20)
  print '%10s\t%3.7f\t%3.7f' % (batch_size, data_amount, real_time)
