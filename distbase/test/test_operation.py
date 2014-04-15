from distbase.cuda_base import relu_activate, tanh_activate, relu_compute_grad, tanh_compute_grad
from distbase.cuda_base import logreg_cost_row_reduce, logreg_cost_col_reduce
from distbase.cuda_base import softmax_bprop
from pycuda import gpuarray, autoinit
import numpy as np
import random
import math

shapes = [(10, 10), (512, 10), (10, 512), (678231, 30), (30, 678231), (1026, 1026), (2049, 1996)]
for fd, sd in shapes:
  print 'build input/output'
  input_local = np.random.randint(10, size = (fd, sd)).astype(np.float32) + 1.0
  output_local = np.zeros((fd, sd)).astype(np.float32)
  ingrad_local = np.random.randn(fd, sd).astype(np.float32)
  outgrad_local = np.zeros((fd, sd)).astype(np.float32)

  input = gpuarray.to_gpu(input_local)
  output = gpuarray.to_gpu(output_local)
  ingrad = gpuarray.to_gpu(ingrad_local)
  outgrad = gpuarray.to_gpu(outgrad_local)

  print 'input.shape', input.shape
  print 'finished'

  relu_activate(input, output, 0)
  output_local[:, :] = input_local[:, :]
  output_local[output_local < 0] = 0;

  diff = np.abs(output_local - output.get())
  assert (diff < 1e-4).all()

  relu_compute_grad(ingrad, output, outgrad, 0) 
  outgrad_local = ingrad_local * (output_local > 0)

  diff = np.abs(outgrad_local - outgrad.get())
  assert (diff < 1e-4).all()
  
  output.fill(0)
  output_local.fill(0)
  outgrad.fill(0)
  outgrad_local.fill(0)
  
  a = random.random()
  b = random.random()
  tanh_activate(input, output, a, b)
  
  output_local =  a * (2  / (np.exp(input_local * (-2.0 * b)) + 1) - 1)
  
  diff = np.abs(output_local - output.get())
  assert (diff < 1e-4).all()

  tanh_compute_grad(ingrad, output, outgrad, a, b)
  tmp = (1.0 - output_local / a) / 2.0
  outgrad_local = tmp * (tmp - 1.0) * (-4.0 * a * b) * ingrad_local
  
  diff = np.abs(outgrad_local - outgrad.get())
  assert (diff < 1e-4).all()

  label_local = np.array([np.random.choice(fd) for i in range(sd)]).astype(np.float32).reshape((1, sd))
  label = gpuarray.to_gpu(label_local)
  
  softmax_bprop(output, label, outgrad)
  outgrad_local = 0 - output_local
  for i in range(input.shape[1]):
    outgrad_local[label_local[0, i], i] += 1

  diff = np.abs(outgrad_local - outgrad.get())
  assert (diff < 1e-4).all()

  cost_local = np.zeros((1, sd)).astype(np.float32)
  cost = gpuarray.to_gpu(cost_local)

  logreg_cost_col_reduce(input, label, cost)
  for i in range(sd):
    cost_local[0, i] = 0 - math.log(input_local[label_local[0, i], i])

  diff = np.abs(cost_local - cost.get())
  assert (diff < 1e-4).all()

  label_local = np.array([np.random.choice(sd) for i in range(fd)]).astype(np.float32).reshape((1, fd))
  label = gpuarray.to_gpu(label_local)
  
  cost_local = np.zeros((1, fd)).astype(np.float32)
  cost = gpuarray.to_gpu(cost_local)

  logreg_cost_row_reduce(input, label, cost)
  for i in range(fd):
    cost_local[0, i] = 0 - math.log(input_local[i, label_local[0, i]])

  diff = np.abs(cost_local - cost.get())
  assert (diff < 1e-4).all()

