import varray as va
import numpy as np
import sys
import time
start = time.time()

assert len(sys.argv) == 2


# input shape should be 96 * 27 * 27 * 128
# output shape shoudl be 96 * 27 * 27 * 128
# define the parameters
size = 3
scale = 0.0001
pow = 0.75
out_shape = (96, 27, 27, 128)

# get the input and filter
import garray as arr
with open('grad') as g, open('input') as i, open('denom') as d, open('output') as o:
  import cPickle as pickle
  input = pickle.load(i)
  grad = pickle.load(g)
  denom = pickle.load(d)
  output = pickle.load(o)
print  'read', time.time() - start

image_y = 27

if sys.argv[1] == 'single':
  print 'single GPU version'
  # single GPU version
  input = arr.array(input, dtype = np.float32)
  grad = arr.array(grad, dtype = np.float32)
  denom = arr.array(denom, dtype = np.float32)
  output = arr.array(output, dtype = np.float32)

  out_grad = arr.reshape_last(arr.zeros(out_shape, dtype = np.float32))

  arr.rnormundo(
      arr.reshape_last(grad),
      arr.reshape_last(denom),
      arr.reshape_last(input),
      arr.reshape_last(output),
      out_grad,
      96, size,image_y, scale, pow)
  print  'rnormundo', time.time() - start

  out_grad = out_grad.reshape(out_shape)
  with open('rnormundo-output-single', 'w') as f:
    pickle.dump(out_grad.get(), f, protocol = -1)
  print  'done with single',time.time() - start

else:
  print 'multi GPU version'

  # get the input and filter again
  import varray as arr
  from varray import rank

  # multi GPU version
  input = arr.square_array(input, slice_dim = (1, 2))
  output = arr.square_array(output, slice_dim = (1, 2))
  denom = arr.square_array(denom, slice_dim = (1, 2))
  grad = arr.square_array(grad, slice_dim = (1, 2))
  out_grad = arr.zeros(out_shape, slice_dim = (1, 2))


  arr.rnormundo(grad, denom, input,output, out_grad,
      96, size, image_y, scale, pow)
  print 'rnormundo', time.time() - start
  out_grad.gather()
  if rank == 0:
    with open('rnormundo-output-multi', 'w') as f:
      pickle.dump(out_grad.local_data.get(), f, protocol = -1)
  print 'done with multi', time.time() - start
