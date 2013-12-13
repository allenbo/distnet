import numpy as np
import os
import garray
multi_gpu = False
if os.environ.get('MULTIGPU', 'no') == 'yes':
  import varray as arr
  multi_gpu = True
  import socket
  print arr.rank, socket.gethostname()
  garray.device_init(arr.rank)
else:
  import garray as arr
  garray.device_init()

def zeros(shape, dtype = np.float32, unique = False):
  if not multi_gpu:
    col = shape[-1]
    row = int(np.prod(shape[:-1]))
    return garray.zeros((row, col), dtype = dtype)
  else:
    return arr.zeros(shape, dtype = dtype, unique = unique)


def allocate(shape, dtype = np.float32, unique = False):
  if not multi_gpu:
    col = shape[-1]
    row = int(np.prod(shape[:-1]))
    return garray.GPUArray((row, col), dtype = dtype)
  else:
    return arr.allocate(shape, dtype, unique = unique)

def convert_shape(shape):
  if not multi_gpu:
    col = shape[-1]
    row = int(np.prod(shape[:-1]))
    return (row, col)
  return shape

def uniformed_array(array, unique = False):
  if not multi_gpu:
    return arr.array(array, dtype = np.float32)
  return arr.array(array, unique = unique, dtype = np.float32)
