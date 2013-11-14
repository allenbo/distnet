from varray.ndarray import VArray, DistMethod, MASTER, rank, WORLD
import numpy as np

def randn(shape, dtype = np.float32, unique = True, slice_method = DistMethod.Stripe, slice_dim = None):
  a = np.random.randn(*shape).astype(dtype)
  return VArray(a, unique, slice_method, slice_dim)

def square_randn(shape, slice_dim, dtype = np.float32):
  a = np.random.randn(*shape).astype(dtype)
  return VArray(a, unique = True, slice_method = DistMethod.Square, slice_dim = slice_dim)


def get_seed():
  if rank == MASTER:
    import time
    seed = int(time.time())
  else:
    seed = 0
  seed = WORLD.bcast(seed, root = MASTER)
  return seed
