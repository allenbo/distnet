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
with open('input') as i, open('grad') as g:
  import cPickle as pickle
  input = pickle.load(i)
  grad = pickle.load(g)
print  'read', time.time() - start


weight_shape = (3, 11, 11, 96)

if sys.argv[1] == 'single':
  print 'single GPU version'
  # single GPU version
  input = arr.array(input, dtype = np.float32)
  grad = arr.array(grad, dtype = np.float32)
  weight_grad = arr.zeros(weight_shape, dtype = np.float32)

  weight_grad = arr.reshape_last(weight_grad)

  arr.wconvolution(
      arr.reshape_last(input),
      arr.reshape_last(grad),
      weight_grad,
      224, 55, 55, 11, padding, stride, 3)
  print  'weight convolution', time.time() - start

  weight_grad = weight_grad.reshape(weight_shape)
  with open('wconv-output-single', 'w') as f:
    pickle.dump(weight_grad.get(), f, protocol = -1)
  print  'done with single',time.time() - start

else:
  print 'multi GPU version'
  # get the input and filter again
  import varray as arr
  from varray import rank

  # multi GPU version
  input = arr.square_array(input, slice_dim = (1, 2))
  grad = arr.square_array(grad, slice_dim = (1, 2))
  weight_grad = arr.zeros(weight_shape, unique = False)
  arr.wconvolution(input, grad, weight_grad,
      224, 55, 55, 11, padding, stride, 3)
  print 'weight convolution', time.time() - start
  if rank == 0:
    with open('wconv-output-multi', 'w') as f:
      pickle.dump(weight_grad.local_data.get(), f, protocol = -1)
  print 'done with multi', time.time() - start
