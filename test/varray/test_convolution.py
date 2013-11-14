import varray as va
import numpy as np
import sys
import time
start = time.time()

assert len(sys.argv) == 2

# define the parameters
padding = 0
stride = 4
out_shape = (96, 55, 55, 128)

# get the input and filter
import garray as arr
with open('filter') as f, open('input') as i:
  import cPickle as pickle
  filter =  pickle.load(f)
  input = pickle.load(i)
print  'read', time.time() - start

print 'the shape of input', input.shape
print 'the shape of filter', filter.shape



if sys.argv[1] == 'single':
  print 'single GPU version'
  # single GPU version
  input = arr.array(input, dtype = np.float32)
  filter = arr.array(filter, dtype = np.float32)
  output = arr.zeros(out_shape, dtype = np.float32)

  output = arr.reshape_last(output)

  arr.convolution(
      arr.reshape_last(input),
      arr.reshape_last(filter),
      output,
      224, 55, 55, padding, stride, 3, 1)
  print  'convolution', time.time() - start

  output = output.reshape(out_shape)
  with open('conv-output-single', 'w') as f:
    pickle.dump(output.get(), f, protocol = -1)
  print  'done with single',time.time() - start

else:
  print 'multi GPU version'

  # get the input and filter again
  import varray as arr
  from varray import rank

  # multi GPU version
  input = arr.square_array(input, slice_dim = (1, 2))
  filter = arr.square_array(filter, slice_dim = (), unique =False)
  output = arr.zeros(out_shape, slice_dim = (1, 2))
  arr.convolution(input, filter, output,
      224, 55, 55, padding, stride, 3, 1)
  print 'convolution', time.time() - start
  output.gather()
  if rank == 0:
    with open('conv-output-multi', 'w') as f:
      pickle.dump(output.local_data.get(), f, protocol = -1)
  print 'done with multi', time.time() - start
