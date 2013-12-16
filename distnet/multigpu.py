import numpy as np
import os
import garray
multi_gpu = False
if os.environ.get('MULTIGPU', 'no') == 'yes':
  import varray as arr
  from varray import DistMethod, rank, size as num_gpu
  multi_gpu = True
  import socket
  print arr.rank, socket.gethostname()
  garray.device_init(arr.rank)
else:
  import garray as arr
  garray.device_init()
  rank = 0
  num_gpu = 1

def zeros(shape, dtype = np.float32, unique = False):
  if not multi_gpu:
    col = shape[-1]
    row = int(np.prod(shape[:-1]))
    return garray.zeros((row, col), dtype = dtype)
  else:
    return arr.zeros(shape, dtype = dtype, unique = unique, slice_dim = 1, slice_method = DistMethod.Stripe)


def allocate(shape, dtype = np.float32, unique = False):
  if not multi_gpu:
    col = shape[-1]
    row = int(np.prod(shape[:-1]))
    return garray.GPUArray((row, col), dtype = dtype)
  else:
    return arr.allocate(shape, dtype, unique = unique, slice_method = DistMethod.Stripe, slice_dim = 1)

def convert_shape(shape):
  if not multi_gpu:
    col = shape[-1]
    row = int(np.prod(shape[:-1]))
    return (row, col)
  return shape

def uniformed_array(array, dtype = np.float32, unique = True, to2dim = False):
  if not multi_gpu:
    return arr.array(array, dtype = dtype, to2dim = to2dim)
  return arr.array(array, unique = unique, dtype = dtype, slice_method = DistMethod.Stripe, slice_dim = 1)
