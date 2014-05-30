import numpy as np
import os
import garray
import math

from garray import ConvDataLayout, FCDataLayout, FilterLayout, WeightLayout
from distbase import util
from distbase.util import issquare

multi_gpu = False

if os.environ.get('MULTIGPU', 'no') == 'yes':
  import varray as arr
  from varray import rank, size as num_gpu
  from varray.context import default_context, Context, CONTEXTMANAGER
  multi_gpu = True
  import socket
  garray.device_init(arr.rank)
  dist_file = 'config/imagenet.cfg.layerdist'
  strategy = util.load(dist_file)
else:
  import garray as arr
  garray.device_init()
  rank = 0
  num_gpu = 1
  strategy = None
  default_context = None
  class _(object):
    pass
  fake_layerdist = _()
  fake_layerdist.global_dist = False
  fake_layerdist.group_state = None
  fake_layerdist.group_size = None
  fake_layerdist.workers_group = []


def build_context(workers_group):
  if default_context is None or len(workers_group) == 1:
    return default_context
  return CONTEXTMANAGER.build_context(workers_group)

def get_layerdist(layer_name):
  if strategy is None:
    return fake_layerdist
  assert layer_name in strategy
  return strategy[layer_name]

def zeros(shape, global_slice_dim = None, group_slice_dim = None, context = default_context):
  if not multi_gpu:
    return garray.zeros(shape, dtype = np.float32)
  else:
    return arr.zeros(shape = shape,
                     global_slice_dim = global_slice_dim,
                     group_slice_dim = group_slice_dim,
                     context = context)

def allocate(shape, global_slice_dim = None, group_slice_dim = None, context = default_context):
  if not multi_gpu:
    return garray.GPUArray(shape, dtype = np.float32)
  else:
    return arr.allocate(shape = shape,
                        global_slice_dim = global_slice_dim,
                        group_slice_dim = group_slice_dim,
                        context = context)

def uniformed_array(array, global_slice_dim = None, group_slice_dim = None, context = default_context, to2dim = False):
  if not multi_gpu:
    return arr.array(array, dtype = np.float32, to2dim = to2dim)
  else:
    return arr.array(array,
                     global_slice_dim = global_slice_dim,
                     group_slice_dim = group_slice_dim,
                     context = context)

def random_uniform(shape, global_slice_dim = None, group_slice_dim = None, context = default_context):
  if not multi_gpu:
    return arr.random_uniform(shape)
  else:
    return arr.random_uniform(shape, global_slice_dim = global_slice_dim,
                                     group_slice_dim = group_slice_dim,
                                     context = default_context)
  
