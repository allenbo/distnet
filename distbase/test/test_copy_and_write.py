from pycuda import gpuarray, autoinit
import numpy as np
import random

from distbase.cuda_base import stride_copy, stride_write, stride_write_sum, gpu_copy_to, gpu_partial_copy_to

for i in range(3):
  shape = []
  for d in range(4):
    shape.append(random.randint(10, 100))
    ha = np.random.randn(*shape).astype(np.float32)

    ga = gpuarray.to_gpu(ha)
    print 'array.shape', ga.shape
    
    inner_count = 10
    for j in range(inner_count):
      slices = []
      new_shape = []
      for k in range(len(shape)):
        fd = random.randint(0, shape[k])
        sd = random.randint(0, shape[k])
        while sd == fd:
          sd = random.randint(0, shape[k])
        new_shape.append(abs(fd - sd))
        a = slice(min(fd, sd), max(fd, sd))
        slices.append(a)

      # copy
      gb = gpuarray.GPUArray(shape = tuple(new_shape), dtype = ga.dtype)
      stride_copy(ga, gb, slices)
      hb = ha[tuple(slices)]
      
      assert (hb == gb.get()).all()


      # write
      hb = np.random.randn(*new_shape).astype(np.float32)
      gb = gpuarray.to_gpu(hb)
      ha[slices] = hb
      
      stride_write(gb, ga, slices)
      assert (ha == ga.get()).all()

      # write_sum, only when the number of dimension is 4
      if len(shape) == 4:
        ha[slices] += hb
        stride_write_sum(gb, ga, slices)
        assert(ha == ga.get()).all()
  print 'finished round %d' % (i + 1)

print 'stride_copy, stride_write, stride_write_sum pass the test'
