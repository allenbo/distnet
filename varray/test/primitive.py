from varray.ndarray import VArray, DistMethod, allocate_like, allocate
from varray import ndarray
import numpy as np
from mpi4py import MPI
import garray
import math
import random
garray.device_init()

random.seed(0)
WORLD = MPI.COMM_WORLD
MASTER = 0
size = WORLD.Get_size()
rank = WORLD.Get_rank()
iteration_count = 4
dlens = [2, 4]

'''
test for sum, max, add, size, get, reshape
'''

def make_shape(d):
  shape = []
  for i in range(d):
    shape.append(random.randint(10, 100))
  return shape

def test_fill():
  for method in [DistMethod.Square, DistMethod.Stripe]:
    for l in dlens:
      shape = tuple(make_shape(l))
      if method == DistMethod.Stripe:
        slice_dim = random.randint(0, len(shape) - 1)
      else:
        start = random.randint(0, len(shape) - 2)
        slice_dim = (start, start + 1)
      va = allocate(shape = shape, slice_method = method, slice_dim = slice_dim)
      scalar = 1
      va.fill(scalar)

      assert (va.local_data.get() == scalar).all()

def test_add():
  ''' The __add__ function only works for FC layer adding bias to output '''
  method = DistMethod.Stripe 
  shape = tuple(make_shape(2))
  array = np.arange(np.prod(shape)).astype(np.float32).reshape(shape)
  slice_dim = 0
  va = VArray(array, unique = False, slice_method = method, slice_dim = slice_dim, local = False)

  bias_shape = (shape[0], 1)
  vb = VArray(shape = bias_shape, unique = False)
  vb.fill(1)
  vc = va + vb
  assert (vc.local_data.get() == va.local_data.get() + vb.local_data.get()).all()

  va = VArray(array, unique = True, slice_method = method, slice_dim = slice_dim, local = False)
  vb = VArray(shape = bias_shape, unique = True, slice_method = method, slice_dim = slice_dim, local = False)
  vb.fill(1)

  vc = va + vb
  assert (vc.local_data.get() == va.local_data.get() + vb.local_data.get()).all()

def test_sub():
  ''' The __sub__ function is only  used in Softmax layer subtracting max to input '''
  method = DistMethod.Stripe
  shape = tuple(make_shape(2))
  array = np.arange(np.prod(shape)).astype(np.float32).reshape(shape)
  slice_dim = 0
  va = VArray(array, unique = False, slice_method = method, slice_dim = slice_dim, local = False)

  bias_shape = (shape[0], 1)
  vb = VArray(shape = bias_shape, unique = False)
  vb.fill(1)
  vc = va - vb

  assert (vc.local_data.get() == va.local_data.get() - vb.local_data.get()).all()

  va = VArray(array, unique = True, slice_method = method, slice_dim = slice_dim, local = False)
  vb = VArray(shape = bias_shape, unique = True, slice_method = method, slice_dim = slice_dim, local = False)
  vb.fill(1)

  vc = va - vb
  assert (vc.local_data.get() == va.local_data.get() - vb.local_data.get()).all()


def test_mul():
  ''' The __mul__ function is only used in FC layer multiplying the mask and output '''
  for unique in [False, True]:
    method = DistMethod.Stripe
    shape = tuple(make_shape(2))
    array = np.arange(np.prod(shape)).astype(np.float32).reshape(shape)
    slice_dim = 0
    va = VArray(array, unique = unique, slice_method = method, slice_dim = slice_dim, local = False)
    vb = allocate_like(va)
    vb.fill(2)
    vc = vb * va
    assert (vc.local_data.get() == vb.local_data.get() * va.local_data.get()).all()

def test_div():
  ''' The __div__ function is only  used in Softmax layer divising sum to output '''
  method = DistMethod.Stripe
  shape = tuple(make_shape(2))
  array = np.arange(np.prod(shape)).astype(np.float32).reshape(shape)
  slice_dim = 0
  va = VArray(array, unique = False, slice_method = method, slice_dim = slice_dim, local = False)

  bias_shape = (shape[0], 1)
  vb = VArray(shape = bias_shape, unique = False)
  vb.fill(2)
  vc = va / vb

  assert (vc.local_data.get() == va.local_data.get() / vb.local_data.get()).all()

  va = VArray(array, unique = True, slice_method = method, slice_dim = slice_dim, local = False)
  vb = VArray(shape = bias_shape, unique = True, slice_method = method, slice_dim = slice_dim, local = False)
  vb.fill(1)

  vc = va / vb
  assert (vc.local_data.get() == va.local_data.get() / vb.local_data.get()).all()

def test_eq():
  ''' The __eq__ function is only used in logreg_cost function '''
  method = DistMethod.Stripe
  shape = tuple(make_shape(2))
  array = np.arange(np.prod(shape)).astype(np.float32).reshape(shape)
  slice_dim = 0
  va = VArray(array, unique = False, slice_method = method, slice_dim = slice_dim, local = False)
  vb = VArray(array, unique = False, slice_method = method, slice_dim = slice_dim, local = False)

  assert (vb == va).local_data.get().all()

  vb.fill(2)
  assert not (vb == va).local_data.get().all()

def test_sum():
  pass


if __name__ == '__main__':
  test_fill()
  test_add()
  test_sub()
  test_mul()
  test_div()
  test_eq()
