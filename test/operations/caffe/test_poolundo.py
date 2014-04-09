import caffe
from pycuda import gpuarray, driver, autoinit
import numpy as np
from distbase.util import divup
from distbase.cuda_base import transpose


for color in [96]:
  print 'color is', color
  batch_size = 128
  image_size = 110
  input_shape = (batch_size, color, image_size, image_size)

  start = 0
  pool_size = 3
  stride = 2

  output_size = divup(image_size - pool_size - start, stride) + 1
  output_shape = (batch_size, color, output_size, output_size)

  input_local = np.random.randn(*input_shape).astype(np.float32)
  outgrad_local = np.random.randn(*input_shape).astype(np.float32)
  output_local = np.random.randn(*output_shape).astype(np.float32)
  ingrad_local = np.random.randn(*output_shape).astype(np.float32)

  output = gpuarray.to_gpu(output_local)
  input = gpuarray.to_gpu(input_local)
  ingrad = gpuarray.to_gpu(ingrad_local)
  outgrad = gpuarray.to_gpu(outgrad_local)

  caffe.convLocalMaxUndo(input, ingrad, output, outgrad, pool_size, start, stride, output_size,
      output_size, image_size)
  driver.Context.synchronize()
  print 'max undo pass the test'
  caffe.convLocalAvgUndo(ingrad, outgrad, pool_size, start, stride, output_size,
      output_size, image_size, image_size)
  driver.Context.synchronize()
  print 'avg undo pass the test'
