import numpy as np
import os
import garray
import math

from garray import ConvDataLayout, FCDataLayout, FilterLayout, WeightLayout
from distbase import util
from distbase.util import issquare, AttrDict

multi_gpu = False

if os.environ.get('MULTIGPU', 'no') == 'yes':
  import varray as arr
  from varray import rank, size as num_gpu
  from varray.context import default_context, Context, CONTEXTMANAGER
  multi_gpu = True
  garray.device_init(3 - arr.rank)
else:
  import garray as arr
  garray.device_init()
  rank, num_gpu = 0, 1
