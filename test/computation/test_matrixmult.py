from pycuda import gpuarray, driver, autoinit
import numpy as np
import time
import garray


input_size = 9216
output_size = 4096

print '%10s\t%10s\t%10s' %('batch', 'comput', 'real')
for batch_size in [8, 16, 32, 64, 128, 256]:
  
  input = gpuarray.to_gpu(np.random.randn(input_size, batch_size).astype(np.float32))
  weight = gpuarray.to_gpu(np.random.randn(output_size, input_size).astype(np.float32))
  output = gpuarray.to_gpu(np.random.randn(output_size, batch_size).astype(np.float32))
  count = 10
  for i in range(count):
    garray.matrixmult(weight, input, dest = output)
    driver.Context.synchronize()

  count = 100
  start = time.time()
  for i in range(count):
    garray.matrixmult(weight, input, dest = output)
    driver.Context.synchronize()

  real_time = (time.time() - start) / count
  data_amount = (input_size * batch_size + output_size * input_size + output_size * batch_size) * 4
  comput_amount = (input_size * batch_size * output_size * 1.0) / (1e9)
  print '%10s\t%3.7f\t%3.7f' % (batch_size, comput_amount, real_time)
