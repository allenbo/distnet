import varray as va
import numpy as np
import sys
import time
start = time.time()

assert len(sys.argv) == 2
def divup(x, base):
  if x / base * base == x:
    return int(x / base)
  else:
    return int(x / base + 1)

# define the parameters

# get the input and filter
import garray as arr
with open('filter') as f, open('input') as i:
  import cPickle as pickle
  filter =  pickle.load(f)
  input = pickle.load(i)
print  'read', time.time() - start

print 'the shape of input', input.shape
print 'the shape of filter', filter.shape

input_c, input_x, _, _ =  input.shape
_, filter_size, _, filter_channel = filter.shape

padding = 1
stride = 2
out_x =  1 + divup(2 * padding + input_x - filter_size, stride)
padding = -padding
out_shape = (filter_channel, out_x, out_x, 128)

print 'output shape', out_shape
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
      input_x, out_x, out_x, padding, stride, input_c, 1)
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

  print 'rank', rank
  # multi GPU version
  input = arr.square_array(input, slice_dim = (1, 2))
  filter = arr.square_array(filter, slice_dim = (), unique =False)
  output = arr.zeros(out_shape, slice_dim = (1, 2))
  arr.convolution(input, filter, output,
      input_x, out_x, out_x, padding, stride, input_c, 1)
  print 'convolution', time.time() - start
  output.gather()
  if rank == 0:
    with open('conv-output-multi', 'w') as f:
      pickle.dump(output.local_data.get(), f, protocol = -1)
  print 'done with multi', time.time() - start
