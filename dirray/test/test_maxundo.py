import varray as va
import numpy as np
import sys
import time
start = time.time()

assert len(sys.argv) == 2


# input shape should be 96 * 55 * 55 * 128
# output shape shoudl be 96 * 27 * 27 * 128
# define the parameters
start = 0
size = 3
stride = 2

# get the input and filter
import garray as arr
with open('grad') as g, open('input') as i, open('output') as o:
  import cPickle as pickle
  input = pickle.load(i)
  grad = pickle.load(g)
  output = pickle.load(o)
print  'read', time.time() - start

out_grad_shape = input.shape

print 'input', input.shape
print 'grad', grad.shape
print 'output', output.shape
image_y = 55
output_y = 27
output_x = 27

if sys.argv[1] == 'single':
  print 'single GPU version'
  # single GPU version
  input = arr.array(input, dtype = np.float32)
  grad = arr.array(grad, dtype = np.float32)
  output = arr.array(output, dtype = np.float32)

  out_grad = arr.reshape_last(arr.zeros_like(input))

  arr.maxundo(
      arr.reshape_last(input),
      arr.reshape_last(grad),
      arr.reshape_last(output),
      out_grad,
      size, start, stride, output_y, output_x, image_y)
  print  'max undo', time.time() - start

  out_grad = out_grad.reshape(out_grad_shape)
  with open('maxundo-output-single', 'w') as f:
    pickle.dump(out_grad.get(), f, protocol = -1)
  print  'done with single',time.time() - start

else:
  print 'multi GPU version'

  # get the input and filter again
  import varray as arr
  from varray import rank

  # multi GPU version
  input = arr.square_array(input, slice_dim = (1, 2))
  grad = arr.square_array(grad, slice_dim = (1, 2))
  output = arr.square_array(output, slice_dim = (1, 2))
  out_grad = arr.zeros(out_grad_shape, slice_dim = (1, 2))
  arr.maxundo(input,grad,output, out_grad,
      size, start, stride, output_y, output_x, image_y)
  print 'maxundo', time.time() - start
  out_grad.gather()
  if rank == 0:
    with open('maxundo-output-multi', 'w') as f:
      pickle.dump(out_grad.local_data.get(), f, protocol = -1)
  print 'done with multi', time.time() - start
