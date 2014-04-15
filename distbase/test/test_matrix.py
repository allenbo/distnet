from pycuda import gpuarray, autoinit
import numpy as np
import random

from distbase.cuda_base import transpose, matrixmult

shapes = [(10, 10), (512, 10), (10, 512), (678231, 30), (30, 678231), (1026, 1026), (2049, 1996)]
for shape in shapes:
  ha = np.random.randn(*shape).astype(np.float32)
  print 'build array'
  ga = gpuarray.to_gpu(ha)
  print 'array.shape', ga.shape

  gb = transpose(ga)
  
  diff = np.abs(np.transpose(ha) - gb.get())
  assert (diff < 1e-5).all()

  print 'transpose pass the test'

  hb_shape = (shape[1], random.randint(1, 2 ** 8))
  hb = np.random.randn(*hb_shape).astype(np.float32)

  gb = gpuarray.to_gpu(hb)
  print 'second.shape', gb.shape
  
  gc = matrixmult(ga, gb)
  hc = np.dot(ha, hb)
  diff = np.abs(gc.get() - hc)/hb_shape[0]
  assert (diff < 1e-5).all()

  alpha = random.random()
  beta = random.random()
  matrixmult(ga, gb, dest = gc, alpha = alpha, beta = beta)

  hc = np.dot(ha, hb) * alpha + hc * beta
  diff = np.abs(gc.get() -hc)/hb.shape[0]
  assert (diff < 1e-5).all()

  print 'matrix pass the test'
