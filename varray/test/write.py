from varray.ndarray import VArray, DistMethod
from varray.ndarray import VArray, DistMethod
from varray.area import Area, Point
import numpy as np
import random
from mpi4py import MPI
import garray
import math
import copy
np.set_printoptions(threshold=np.nan)
garray.device_init()

'''
write_local, write_remote, write
'''
random.seed(0)
WORLD = MPI.COMM_WORLD
MASTER = 0
size = WORLD.Get_size()
rank = WORLD.Get_rank()

dlens = [2, 4]
iteration_count = 4

def make_shape(d):
  shape = []
  for i in range(d):
    shape.append(random.randint(40, 100))
  return shape

def make_area_in(area):
  length = len(area._from)
  _from = area._from
  _to = area._to

  f = []
  t = []
  for i in range(length):
    f.append(random.randint(_from[i], _to[i] - 2))
    t.append(random.randint(f[i],  _to[i]))
  
  return Area(Point(*f), Point(*t))


def make_local_area(va):
  return make_area_in(va.local_area)

def test_write_local():
  for i in range(iteration_count):
    for method in [DistMethod.Square, DistMethod.Stripe]:
      for l in dlens:
        shape = tuple(make_shape(l))
        array = np.ones(shape).astype(np.float32)
        # square
        if method == DistMethod.Stripe:
          slice_dim = random.randint(0, len(shape) - 1)
        else:
          start = random.randint(0, len(shape) - 2)
          slice_dim = (start, start + 1)
        va = VArray(array, unique = True, slice_method = method, slice_dim = slice_dim, local = False)
        
        chunk_areas = [va.local_area]
        for _ in range(3):
          chunk_areas.append(make_local_area(va))
        for chunk_area in chunk_areas:
          data = garray.array(np.ones(shape = chunk_area.shape).astype(np.float32))
          if l == 4:
            incr = va.fetch_local(chunk_area)
            if incr is va.local_data:
              incr = garray.array(va.local_data.get())
            va.write_local(chunk_area, incr, acc = True)
            assert ((incr * 2).get() == va.fetch_local(chunk_area).get()).all()
          
          va.write_local(chunk_area, data)
          assert (data.get() == va.fetch_local(chunk_area).get()).all()
          
def test_write_remote():
  ''' only works for 4D array '''
  for i in range(iteration_count):
    for method in [DistMethod.Square, DistMethod.Stripe]:
      shape = tuple(make_shape(4))
      array = np.ones(shape).astype(np.float32)
      # square
      if method == DistMethod.Stripe:
        slice_dim = random.randint(0, len(shape) - 1)
      else:
        start = random.randint(0, len(shape) - 2)
        slice_dim = (start, start + 1)
      va = VArray(array, unique = True, slice_method = method, slice_dim = slice_dim, local = False)
      
      reqs = [None] * size
      local_subs = [None] * size
      for s in range(size):
        chunk_area = make_area_in(va.area_dict[s])
        chunk_data = np.ones(shape = chunk_area.shape).astype(np.float32)
        if s == rank:
          local_area = chunk_area
          local_data = chunk_data
        else:
          reqs[s] = chunk_area
          local_subs[s] = garray.array(chunk_data)
      
      va.write_remote(reqs, local_subs)
      local_data *= (size-1)
      array[local_area.slice] += local_data
      assert (array[va.local_area.slice] == va.local_data.get()).all()


def test_write():
  for i in range(iteration_count):
    for method in [DistMethod.Square, DistMethod.Stripe]:
      shape = tuple(make_shape(4))
      array = np.ones(shape).astype(np.float32)
      if method == DistMethod.Stripe:
        slice_dim = random.randint(0, len(shape) - 1)
      else:
        start = random.randint(0, len(shape) - 2)
        slice_dim = (start, start + 1)
      va = VArray(array, unique = True, slice_method = method, slice_dim = slice_dim, local = False)

      for _ in range(7):
        chunk_area = make_area_in(va.global_area)
        chunk_data = garray.array(np.ones(shape = chunk_area.shape).astype(np.float32))

        va.write(chunk_area, chunk_data)

        array[chunk_area.slice] = size * chunk_data.get()
        assert (array[va.local_area.slice] == va.local_data.get()).all()

        va.write(chunk_area, chunk_data, propagate = False)
        array[chunk_area.slice] = chunk_data.get()
        assert (array[va.local_area.slice] == va.local_data.get()).all()
  # create two different varray    
  shape = tuple(make_shape(4))
  array = np.ones(shape).astype(np.float32)

  va = VArray(array * 2, unique = True, slice_method = DistMethod.Square, slice_dim = (1, 2), local = False)
  vb = VArray(array, unique = True, slice_method = DistMethod.Stripe, slice_dim = 3, local = False)
  vb.fill(0)
  vb.write(area = va.local_area, data = va.local_data, propagate = True, debug = True)
  assert (vb.local_data.get() == (array[vb.local_area.slice] * 2)).all()

if __name__ == '__main__':
  test_write_local()
  test_write_remote()
  test_write()
