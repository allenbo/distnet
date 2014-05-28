from .ndarray import *
from .area import *
from .operations import *

def get_seed():
  if rank == MASTER:
    import time
    seed = int(time.time())
  else:
    seed = 0
  seed = WORLD.bcast(seed, root = MASTER)
  return seed

def distlog(_fn):
  if rank == MASTER:
    return _fn
  else:
    return lambda msg, *arg, **kw : 0
