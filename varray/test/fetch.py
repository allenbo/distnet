from varray.ndarray import VArray, DistMethod
from varray.ndarray import VArray, DistMethod
from varray.area import Area, Point
import numpy as np
import random
from mpi4py import MPI
import garray
import math
import copy
garray.device_init()

'''
fetch_local, fetch_remote, fetch, gather, merge
'''
WORLD = MPI.COMM_WORLD
MASTER = 0
size = WORLD.Get_size()
rank = WORLD.Get_rank()

dlens = [2, 4]
iteration_count = 4

def make_shape(d):
  shape = []
  for i in range(d):
    shape.append(random.randint(10, 100))
  return shape

def make_area_in(area):
  length = len(area._from)
  _from = area._from
  _to = area._to

  f = []
  t = []
  for i in range(length):
    f.append(random.randint(_from[i], _to[i] - 1))
    t.append(random.randint(f[i],  _to[i]))
  
  return Area(Point(*f), Point(*t))


def make_local_area(va):
  return make_area_in(va.local_area)

def make_corner_area(va, d, slice_dim, padding):
  assert 0 <= d <= 3
  row, col = slice_dim
  if d == 0:
    _from = va.area_dict[0]._from[:]
    _to = va.area_dict[0]._to[:]

    _to[row] += padding
    _to[col] += pading
    return Area(Point(*_from), Point(*_to))
  
  if d == 1:
    _from = va.area_dict[int(math.sqrt(size)) - 1]._from[:]
    _to = va.area_dict[int(math.sqrt(size)) - 1]._to[:]

    _from[col] -= padding
    _to[row] += padding
    return Area(Point(*_from), Point(*_to))

  if d == 2:
    _from = va.area_dict[size - 1]._from[:]
    _to = va.area_dict[size - 1]._to[:]

    _from[col] -= padding
    _to[row] -= padding
    return Area(Point(*_from), Point(*_to))

  if d == 3:
    _from = va.area_dict[size - int(math.sqrt(size))]._from[:]
    _to = va.area_dict[size - int(math.sqrt(size))]._to[:]

    _from[row] -= padding
    _to[col] += padding
    return Area(Point(*_from), Point(*_to))


def test_fetch_local():
  for i in range(iteration_count):
    for method in [DistMethod.Square, DistMethod.Stripe]:
      for l in dlens:
        shape = tuple(make_shape(l))
        array = np.arange(np.prod(shape)).astype(np.float32).reshape(shape)
        # square
        if method == DistMethod.Stripe:
          slice_dim = random.randint(0, len(shape) - 1)
        else:
          start = random.randint(0, len(shape) - 2)
          slice_dim = (start, start + 1)
        va = VArray(array, unique = True, slice_method = method, slice_dim = slice_dim, local = False)

        data = va.fetch_local(None)
        assert data is None
        data = va.fetch_local(va.local_area)
        assert data is va.local_data
        
        local_area = make_local_area(va) 
        assert local_area in va.local_area
        data = va.fetch_local(local_area)
        assert (data.get() == array[local_area.slice]).all()


def test_fetch_remote():
  for i in range(iteration_count):
    for method in [DistMethod.Square, DistMethod.Stripe]:
      for l in dlens:
        shape = tuple(make_shape(l))
        array = np.arange(np.prod(shape)).astype(np.float32).reshape(shape)
        # square
        if method == DistMethod.Stripe:
          slice_dim = random.randint(0, len(shape) - 1)
        else:
          start = random.randint(0, len(shape) - 2)
          slice_dim = (start, start + 1)
        va = VArray(array, unique = True, slice_method = method, slice_dim = slice_dim, local = False)
        
        reqs = [None] * size

        for s in range(size):
          if s != rank:
            reqs[s] = make_area_in(va.area_dict[s])

        subs = va.fetch_remote(reqs)
        for sub_area, sub_data in subs.iteritems():
          if sub_area is not None and sub_data is not None:
            assert (sub_data.get() == array[sub_area.slice]).all()


def test_pad():
  ''' test for get_pad_info '''
  for i in range(iteration_count):
    for l in dlens:
      shape = tuple(make_shape(l))
      array = np.arange(np.prod(shape)).astype(np.float32).reshape(shape)
      # square
      method = DistMethod.Square
      start = random.randint(0, len(shape) - 2)
      slice_dim = (start, start + 1)

      va = VArray(array, unique = True, slice_method = method, slice_dim = slice_dim, local = False)
      
      padding = 1
      for j in range(size):
        chunk_area = va.area_dict[j]
        new_shape, slices = va.get_pad_info(padding, chunk_area.shape, chunk_area, slice_dim = slice_dim)
        cal_shape = list(chunk_area.shape)

        row, col = slice_dim

        diff_slice = [0] * l
        if chunk_area._from[row] == 0:
          cal_shape[row] += padding
          diff_slice[row] += padding
        if chunk_area._from[col] == 0:
          cal_shape[col] += padding
          diff_slice[col] += padding

        if chunk_area._to[row] == va.global_area._to[row]:
          cal_shape[row] += padding
        if chunk_area._to[col] == va.global_area._to[col]:
          cal_shape[col] += padding

        assert cal_shape == list(new_shape)
        assert diff_slice == [x.start for x in slices]
      
def test_merge():
  for i in range(iteration_count):
    for method in [DistMethod.Square, DistMethod.Stripe]:
      for l in dlens:
        shape = tuple(make_shape(l))
        array = np.arange(np.prod(shape)).astype(np.float32).reshape(shape)
        # square
        if method == DistMethod.Stripe:
          slice_dim = random.randint(0, len(shape) - 1)
        else:
          start = random.randint(0, len(shape) - 2)
          slice_dim = (start, start + 1)
        va = VArray(array, unique = True, slice_method = method, slice_dim = slice_dim, local = False)
        
        chunk_areas = []
        chunk_areas.append(make_area_in(va.global_area))
        if method == DistMethod.Square:
          padding = -1
          for d in range(4):
            chunk_areas.append(make_corner_area(va, d, slice_dim))
        
        for chunk_area in chunk_areas:
          subs = {}
          reqs = [None] * size
          if chunk_area in va.local_area:
            subs[chunk_area] = va.fetch_local(chunk_area)
            for i in range(size):
              if i != rank:
                reqs[i] = None
          else:
            for r, a in va.area_dict.iteritems():
              sub_area = a & chunk_area
              if r == rank:
                sub_array = va.fetch_local(sub_area)
                subs[sub_area] = sub_array
                reqs[r] = None
              else:
                reqs[r] = sub_area
          subs.update(va.fetch_remote(reqs))
          data = va.merge(subs, chunk_area)
          assert (data.get() == array[chunk_area.slice]).all()

          if method == DistMethod.Square:
            ispadding = random.randint(0, 1) == 1
            rpadding = padding if ispadding else 0
            data = va.merge(subs, chunk_area, padding = rpadding, slice_dim = slice_dim)
            _, slices = va.get_pad_info(-rpadding, chunk_area.shape, chunk_area, slice_dim)
            assert (data[slices].get() == array[chunk_area.slice]).all()
          

def test_fetch():
  for i in range(iteration_count):
    for method in [DistMethod.Square, DistMethod.Stripe]:
      for l in dlens:
        shape = tuple(make_shape(l))
        array = np.arange(np.prod(shape)).astype(np.float32).reshape(shape)
        # square
        if method == DistMethod.Stripe:
          slice_dim = random.randint(0, len(shape) - 1)
        else:
          start = random.randint(0, len(shape) - 2)
          slice_dim = (start, start + 1)
        va = VArray(array, unique = True, slice_method = method, slice_dim = slice_dim, local = False)
        
        chunk_areas = []
        for _  in range(3):
          chunk_areas.append(make_area_in(va.global_area))
        if method == DistMethod.Square:
          padding = -1
          for d in range(4):
            chunk_areas.append(make_corner_area(va, d, slice_dim))

        for chunk_area in chunk_areas:
          if method == DistMethod.Square:
            ispadding = random.randint(0, 1) == 1
            rpadding = padding if ispadding else 0

            data = va.fetch(chunk_area, padding = rpadding, slice_dim = slice_dim)
            
            new_shape, slices = va.get_pad_info(-rpadding, chunk_area.shape, chunk_area, slice_dim)
            assert (data[slices].get() == array[chunk_area.slice]).all()
          else:
            data = va.fetch(chunk_area)
            assert (data.get() == array[chunk_area.slice]).all()

def test_gather():
  for i in range(iteration_count):
    for method in [DistMethod.Square, DistMethod.Stripe]:
      for l in dlens:
        shape = tuple(make_shape(l))
        array = np.arange(np.prod(shape)).astype(np.float32).reshape(shape)
        # square
        if method == DistMethod.Stripe:
          slice_dim = random.randint(0, len(shape) - 1)
        else:
          start = random.randint(0, len(shape) - 2)
          slice_dim = (start, start + 1)
        va = VArray(array, unique = True, slice_method = method, slice_dim = slice_dim, local = False)
        
        va.gather()
        assert (va.local_data.get() == array).all()
 
if __name__ == '__main__':
  test_fetch_local()
  test_fetch_remote()
  test_pad()
  test_merge()
  test_fetch()
  test_gather()
