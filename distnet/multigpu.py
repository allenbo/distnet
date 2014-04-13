import numpy as np
import os
import garray
import math
from garray import ConvDataLayout, FCDataLayout, FilterLayout, WeightLayout

from distbase import util
from distbase.state import *

multi_gpu = False

if os.environ.get('MULTIGPU', 'no') == 'yes':
  import varray as arr
  from varray import DistMethod, rank, size as num_gpu
  multi_gpu = True
  import socket
  print arr.rank, socket.gethostname()
  garray.device_init(arr.rank)
  dist_file = 'distribution/strategy'
  strategy = util.load(dist_file)
else:
  import garray as arr
  garray.device_init()
  rank = 0
  num_gpu = 1
  strategy = None

def get_state(layer_name):
  if strategy is None:
    return None
  return strategy.get('layer_name', None)

def issquare(x):
  a = int(math.sqrt(x))
  if a ** 2 == x:
    return True
  else:
    return False

def zeros(shape, dtype = np.float32, unique = False, slice_dim = None):
  if not multi_gpu:
    return garray.zeros(shape, dtype = dtype)
  else:
    if issquare(num_gpu):
      slice_method = DistMethod.Square
      if slice_dim is not None:
        if np.isscalar(slice_dim):
          slice_method = DistMethod.Stripe
        else:
          slice_method = DistMethod.Square
        unique = True
      else:
        unique = False
      return arr.zeros(shape, dtype, unique = unique, slice_method = slice_method, slice_dim = slice_dim)
    else:
      if slice_dim is not None:
        assert np.isscalar(slice_dim)
        unique = True
      else:
        unique = False
      return arr.zeros(shape, dtype = dtype, unique = unique, slice_dim = slice_dim, slice_method = DistMethod.Stripe)


def allocate(shape, dtype = np.float32, slice_dim = None):
  if not multi_gpu:
    return garray.GPUArray(shape, dtype = dtype)
  else:
    if issquare(num_gpu):
      slice_method = DistMethod.Square
      if slice_dim is not None:
        if np.isscalar(slice_dim):
          slice_method = DistMethod.Stripe
        else:
          slice_method = DistMethod.Square
        unique = True
      else:
        unique = False
      return arr.allocate(shape, dtype, unique = unique, slice_method = slice_method, slice_dim = slice_dim)
    else:
      if slice_dim is not None:
        assert np.isscalar(slice_dim)
        unique = True
      else:
        unique = False
      return arr.allocate(shape, dtype, unique = unique, slice_method = DistMethod.Stripe, slice_dim = slice_dim)

def convert_shape(shape):
  if not multi_gpu:
    col = shape[-1]
    row = int(np.prod(shape[:-1]))
    return (row, col)
  return shape

def uniformed_array(array, dtype = np.float32, slice_dim = None, to2dim = False):
  if not multi_gpu:
    return arr.array(array, dtype = dtype, to2dim = to2dim)
  if issquare(num_gpu):
    slice_method = DistMethod.Square
    if slice_dim is not None:
      if np.isscalar(slice_dim):
        slice_method = DistMethod.Stripe
      else:
        slice_method = DistMethod.Square
      unique = True
    else:
      unique = False
    return arr.array(shape, dtype, unique = unique, slice_method = slice_method, slice_dim = slice_dim)
  else:
    if slice_dim is not None:
      assert np.isscalar(slice_dim)
      unique = True
    else:
      unique = False
    return arr.array(shape, dtype, unique = unique, slice_method = DistMethod.Stripe, slice_dim = slice_dim)
