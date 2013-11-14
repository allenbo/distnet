import varray as va
import numpy as np
import sys
import time
start = time.time()

assert len(sys.argv) == 2

# define the parameters
padding = 0
stride = 4

# get the input and filter
import garray as arr
with open('filter') as f, open('input') as i, open('grad') as g:
  import cPickle as pickle
  filter =  pickle.load(f)
  input = pickle.load(i)
  grad = pickle.load(g)
print  'read', time.time() - start

print 'the shape of input', input.shape
print 'the shape of filter', filter.shape


out_grad_shape = input.shape

if sys.argv[1] == 'single':
  print 'single GPU version'
  # single GPU version
  input = arr.array(input, dtype = np.float32)
  filter = arr.array(filter, dtype = np.float32)
  grad = arr.array(grad, dtype = np.float32)
  out_grad = arr.zeros(out_grad_shape, dtype = np.float32)

  out_grad = arr.reshape_last(out_grad)

  print 'input', input.shape
  print 'filter', filter.shape
  print 'grad', grad.shape
  print 'out_grad', out_grad.shape
  arr.bconvolution(
      arr.reshape_last(input),
      arr.reshape_last(grad),
      arr.reshape_last(filter),
      out_grad,
      224, 224, 55, padding, stride, 3)
  print  'back convolution', time.time() - start

  out_grad = out_grad.reshape(out_grad_shape)
  with open('bconv-output-single', 'w') as f:
    pickle.dump(out_grad.get(), f, protocol = -1)
  print  'done with single',time.time() - start

else:
  print 'multi GPU version'
  # get the input and filter again
  import varray as arr
  from varray import rank

  # multi GPU version
  input = arr.square_array(input, slice_dim = (1, 2))
  filter = arr.square_array(filter, slice_dim = (), unique =False)
  grad = arr.square_array(grad, slice_dim = (1, 2))
  out_grad = arr.zeros(out_grad_shape, slice_dim = (1, 2))
  arr.bconvolution(input, grad, filter, out_grad,
      224, 224, 55, padding, stride, 3)
  print 'bconvolution', time.time() - start
  out_grad.gather()
  if rank == 0:
    with open('bconv-output-multi', 'w') as f:
      pickle.dump(out_grad.local_data.get(), f, protocol = -1)
  print 'done with multi', time.time() - start
