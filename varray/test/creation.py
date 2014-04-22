from varray.ndarray import VArray, DistMethod
from varray import ndarray
import numpy as np
import random
from mpi4py import MPI
import garray
import math
garray.device_init()

random.seed(0)
WORLD = MPI.COMM_WORLD
MASTER = 0
size = WORLD.Get_size()
rank = WORLD.Get_rank()


'''
from_stripe
'''

def make_shape(d):
  shape = []
  for i in range(d):
    shape.append(random.randint(10, 300))
  return shape

def check_square(va, array, shape):
  assert va.rank == rank
  assert va.world_size == size
  assert va.unique == True
  assert va.shape == tuple(shape)
  assert va.slice_method == DistMethod.Square
  assert len(va.slice_dim) == 2
  assert va.nprow == int(math.sqrt(size))
  if array is None: return
  if isinstance(array, garray.GPUArray):
    assert (array.get()[va.local_area.slice] == va.local_data.get()).all()
  else:
    assert (array[va.local_area.slice] == va.local_data.get()).all()

def check_stripe(va, array, shape):
  assert va.rank == rank
  assert va.world_size == size
  assert va.unique == True
  assert va.shape == tuple(shape)
  assert va.slice_method == DistMethod.Stripe
  assert va.slice_dim == 0
  
  if array is None: return
  if isinstance(array, garray.GPUArray):
    assert (array.get()[va.local_area.slice] == va.local_data.get()).all()
  else:
    assert (array[va.local_area.slice] == va.local_data.get()).all()

def check_share(va, array, shape):
  assert va.rank == rank
  assert va.world_size == size
  assert va.unique == False
  assert va.shape == tuple(shape)
  assert va.local_shape == tuple(shape)
  assert va.slice_dim is None
  assert va.slice_method is None
  if array is None: return
  if isinstance(array, garray.GPUArray):
    assert (array.get() == va.local_data.get()).all()
  else:
    assert (array == va.local_data.get()).all()

def test_square():
  shape = make_shape(2)
  # regular instance
  array = np.random.randn(*shape).astype(np.float32)
  va = VArray(array, unique = True, slice_method = DistMethod.Square, slice_dim = (0, 1),local = False)
  check_square(va, array, shape)

  # garray instance
  array = garray.array(array)
  va = VArray(array, unique = True, slice_method = DistMethod.Square, slice_dim = (0, 1), local = False)
  check_square(va, array, shape)
  
   
  # local instance, the array parameter should not be global
  va = VArray(unique = True, slice_method = DistMethod.Square, slice_dim = (0, 1), shape = tuple(shape), local = False)
  check_square(va, None, shape)
  
  # local instance with array
  new_shape = [x * int(math.sqrt(size)) for x in shape] 
  va = VArray(array, unique = True, slice_method = DistMethod.Square, slice_dim = (0, 1), shape = tuple(new_shape), local = True)

  check_square(va, None, new_shape)
  assert (array.get() == va.local_data.get()).all()

def test_stripe():
  shape = make_shape(2)
  # regular instance
  array = np.random.randn(*shape).astype(np.float32)
  va = VArray(array, unique = True, slice_method = DistMethod.Stripe, slice_dim = 0, local = False)
  check_stripe(va, array, shape)

  # garray instance
  array = garray.array(array)
  va = VArray(array, unique = True, slice_method = DistMethod.Stripe, slice_dim = 0, local = False)
  check_stripe(va, array, shape)

  # local instance, the array parameter should not be global
  va = VArray(unique = True, slice_method = DistMethod.Stripe, slice_dim = 0, shape = tuple(shape), local = False)
  check_stripe(va, None, shape) 
  
  # local instance with array
  shape[0] *= int(size)
  va = VArray(array, unique = True, slice_method = DistMethod.Stripe, slice_dim = 0, shape = tuple(shape), local = True)
  check_stripe(va, None, shape)
  assert (array.get() == va.local_data.get()).all()


def test_share():
  shape = make_shape(2)  
  array = np.random.randn(*shape).astype(np.float32)

  va = VArray(array, unique = False)
  check_share(va, array, shape)
  

  va = VArray(unique = False, shape = shape)
  check_share(va, None, shape)


def test_array():
  shape = make_shape(2)
  slice_method = DistMethod.Square
  slice_dim = (0, 1)

  array = np.random.randn(*shape).astype(np.float32)
  va = ndarray.array(array, slice_method = slice_method, slice_dim = slice_dim)
  check_square(va, array, shape)

  slice_method = DistMethod.Stripe
  slice_dim = random.randint(0, 1)
  va = VArray(array, slice_method = slice_method, slice_dim = slice_dim)
  check_stripe(va, array, shape)
 
def test_square_array():
  shape = make_shape(2)
  slice_dim = (0, 1)

  array = np.random.randn(*shape).astype(np.float32)
  va = ndarray.square_array(array, slice_dim = slice_dim)
  check_square(va, array, shape)


def test_zeros():
  shape = make_shape(2)
  slice_method = DistMethod.Square
  slice_dim = (0, 1)
  va = ndarray.zeros(shape, slice_method = slice_method, slice_dim = slice_dim)
  check_square(va, None, shape)
  assert (va.local_data.get() == 0).all()

  slice_method = DistMethod.Stripe
  slice_dim = random.choice([0, 1])
  va = ndarray.zeros(shape, slice_method = slice_method, slice_dim = slice_dim)
  check_stripe(va, None, shape)
  assert (va.local_data.get() == 0).all()

def test_from_stripe_to_square():
  shape = make_shape(2)
  slice_dim = (0, 1)
  axis = 1
  
  array = np.random.randn(*shape).astype(np.float32)
  va = ndarray.from_stripe_to_square(array, slice_dim = slice_dim, axis = axis)
  new_shape = shape[:]
  new_shape[axis] *= size
  check_square(va, None, new_shape)

  axis = 0
  va = ndarray.from_stripe_to_square(array, slice_dim = slice_dim, axis = axis)
  new_shape = shape[:]
  new_shape[axis] *= size
  check_square(va, None, new_shape)


if __name__ == '__main__':
  test_square()
  test_stripe()
  test_share() 
  test_array()
  test_square_array()
  test_from_stripe_to_square()
