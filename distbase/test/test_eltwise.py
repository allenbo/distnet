from pycuda import gpuarray, autoinit
import numpy as np
import random

from distbase.cuda_base import eltwise_mul, eltwise_exp, bigger_than_scalar

shapes = [(10, 10), (512, 10), (10, 512), (678231, 30), (30, 678231), (1026, 1026), (2049, 1996)]
for fd, sd in shapes:
  print 'build src/dest(right)'
  src_local = np.random.randn(fd, sd).astype(np.float32)
  dest_local = np.zeros((fd, sd)).astype(np.float32)
  right_local = np.random.randn(fd, sd).astype(np.float32)

  src = gpuarray.to_gpu(src_local)
  dest = gpuarray.to_gpu(dest_local)
  right = gpuarray.to_gpu(right_local) 

  print 'src.shape', src.shape
  print 'finished'
  
  eltwise_mul(src, right, dest)
  dest_local = src_local * right_local
  
  diff = np.abs(dest_local - dest.get())
  assert (diff < 1e-4).all()

  dest.fill(0)
  dest_local.fill(0)
  eltwise_exp(src, dest)
  dest_local = np.exp(src_local)

  diff = np.abs(dest_local - dest.get())
  assert (diff < 1e-4).all()


  dest.fill(0)
  dest_local.fill(0)
  bigger_than_scalar(src, 0.5, dest)
  dest_local[src_local >= 0.5] = 1.0

  diff = np.abs(dest_local - dest.get())
  assert (diff < 1e-4).all()
