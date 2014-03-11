from pycuda import gpuarray, driver, autoinit
import numpy as np
import time
import garray


batch_size = 128
for batch_size in [32, 64, 128, 256]:
  input_size = 9216
  output_size = 4096

  input = gpuarray.to_gpu(np.random.randn(input_size, batch_size).astype(np.float32))
  weight = gpuarray.to_gpu(np.random.randn(output_size, input_size).astype(np.float32))
  output = gpuarray.to_gpu(np.random.randn(output_size, batch_size).astype(np.float32))
  print 'batch_size = %d' % batch_size
  count = 10
  for i in range(count):
    garray.matrixmult(weight, input, dest = output)
    driver.Context.synchronize()

  count = 100
  start = time.time()
  for i in range(count):
    garray.matrixmult(weight, input, dest = output)
    driver.Context.synchronize()

  print '%f seconds for per matrixmult' % ( (time.time() - start) / count)
  data_amount = (input_size * batch_size + output_size * input_size + output_size * batch_size) * 4
  comput_amount = input_size * batch_size * output_size
  print 'load [%fMB] data' % (data_amount * 1.0 / (1 << 20))
  print 'do [%G] float multiplication' % (comput_amount * 1.0 / 1e9)
