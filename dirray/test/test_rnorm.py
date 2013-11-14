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
with open('filter') as f, open('input') as i:
  import cPickle as pickle
  input = pickle.load(i)
print  'read', time.time() - start

image_y = 27

if sys.argv[1] == 'single':
  print 'single GPU version'
  # single GPU version
  input = arr.array(input, dtype = np.float32)
  output = arr.zeros(out_shape, dtype = np.float32)
  denom = arr.zeros(out_shape, dtype = np.float32)

  output = arr.reshape_last(output)
  denom = arr.reshape_last(denom)

  arr.rnorm(
      arr.reshape_last(input),
      denom, output,
      96, size,image_y, scale, pow)
  print  'rnorm', time.time() - start

  output = output.reshape(out_shape)
  denom = denom.reshape(out_shape)
  with open('rnorm-output-single', 'w') as f:
    pickle.dump(output.get(), f, protocol = -1)
  with open('denom-output-single', 'w') as f:
    pickle.dump(denom.get(), f, protocol = -1)
  print  'done with single',time.time() - start

else:
  print 'multi GPU version'

  # get the input and filter again
  import varray as arr
  from varray import rank

  # multi GPU version
  input = arr.square_array(input, slice_dim = (1, 2))
  output = arr.zeros(out_shape, slice_dim = (1, 2))
  denom = arr.zeros(out_shape, slice_dim = (1, 2))
  arr.rnorm(input,denom, output,
      96, size, image_y, scale, pow)
  print 'rnorm', time.time() - start
  output.gather()
  denom.gather()
  if rank == 0:
    with open('rnorm-output-multi', 'w') as f:
      pickle.dump(output.local_data.get(), f, protocol = -1)
    with open('denom-output-multi', 'w') as f:
      pickle.dump(denom.local_data.get(), f, protocol = -1)
  print 'done with multi', time.time() - start
