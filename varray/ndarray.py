from mpi4py import MPI
from varray.area import Area, Point
import util
from util import issquare
import numpy as np
import math
import copy
import garray

WORLD = MPI.COMM_WORLD
MASTER = 0
size = WORLD.Get_size()
rank = WORLD.Get_rank()

class DistMethod(object):
  Square = 'square'
  Stripe = 'stripe'



class VArray(object):
  def __init__(self, array, unique = True,
                            slice_method = DistMethod.Square,
                            slice_dim = None):
    self.rank = rank
    self.size = size

    self.unique = unique
    self.slice_method = slice_method
    self.slice_dim = slice_dim

    self.area_dict = {}

    if hasattr(array, 'shape'):
      self.global_shape = array.shape
    else:
      assert False, 'Array has no shape attr'

    if hasattr(array, 'dtype'):
      self.dtype = array.dtype
    else:
      assert False, 'Array has no dtype attr'

    if not self.unique:
      if isinstance(array, garray.GPUArray):
        self.local_data = array
      else:
        self.local_data = garray.array(array)
      self.local_area = Area.make_area(self.local_shape)
    else:
      if self.slice_method is None:
        self.slice_method = DistMethod.Square

      if self.slice_method == DistMethod.Square:
        assert issquare(self.size), 'The size of MPI processes has to square'
        assert slice_dim, 'Must specify slice_diim'
        assert len(slice_dim) == 2, 'Length of slice_dim must be 2'

        self.nprow = math.sqrt(self.size)

        self.local_area = self.make_square_area()
        self.local_data = garray.array(array.__getitem__(self.local_area.slice).copy())
      elif self.slice_method == DistMethod.Stripe:
        if self.slice_dim is None:
          self.slice_dim = 0
        else:
          assert np.isscalar(self.slice_dim), 'SLice dim has to the a scalar'

        self.local_area = self.make_stripe_area()
        self.local_data = garray.array(array.__getitem__(self.local_area.slice).copy())
      else:
        assert False, 'No implementation'

    self.area_dict[self.rank] = self.local_area
    self.sync_area_dict()

  def _log(self, fmt, *args):
    util.D('%s :: %s' % (self.rank, fmt % args))

  def sync_area_dict(self):
    rev = WORLD.allgather(self.area_dict[self.rank])
    for i in range(self.size):
      self.area_dict[i] = rev[i]

  @property
  def shape(self):
    return self.global_shape

  @property
  def global_area(self):
    return Area.make_area(self.global_shape)

  def gather(self):
    if not self.unique:
      return

    self.unique = False
    global_area = self.global_area

    self.local_data = self.fetch(global_area)
    self.local_data = garray.array(np.require(self.local_data, dtype = np.float32, requirements = 'C'))
    self.local_area = global_area

    self.area_dict[self.rank] = self.local_area
    self.sync_area_dict()

  def fetch_local(self, area):
    if area is None:
      return None
    area = area.offset(self.local_area._from)
    data = self.local_data.get().__getitem__(area.slice)
    return data

  def fetch_remote(self, reqs):
    subs = {}
    req_list = reqs[:]
    req_list = WORLD.alltoall(req_list)

    send_data = [self.fetch_local(req_list[rank]) for rank in range(self.size)]
    send_data = WORLD.alltoall(send_data)
    WORLD.barrier()
    subs = { reqs[rank]: send_data[rank] for rank in range(self.size)}
    return subs

  def fetch(self, area):
    subs = {}
    reqs = [None] * self.size
    if area in self.local_area:
      subs[area] = self.fetch_local(area)
      for i in range(self.size):
        if i != self.rank:
          reqs[i] = None
    else:
      for rank, a in self.area_dict.iteritems():
        sub_area = a & area
        if rank == self.rank:
          sub_array = self.fetch_local(sub_area)
          subs[sub_area] = sub_array
        reqs[rank] = sub_area
    subs.update(self.fetch_remote(reqs))
    return self.merge(subs, area)


  def write_local(self, area,  data, acc = 'overwrite'):
    if area is None:
      return
    area = area.offset(self.local_area._from)
    cpu_data = self.local_data.get()
    if not isinstance(data, np.ndarray):
      data = data.get()
    if acc == 'overwrite':
      cpu_data.__setitem__(area.slice, data)
    elif acc == 'add':
      cpu_data.__setitem__(area.slice, cpu_data.__getitem__(area.slice) + data)
    else:
      assert False
    self.local_data = garray.array(np.require(cpu_data, dtype = np.float32, requirements = 'C'))



  def write_remote(self, reqs, sub_data, acc):
    req_list = reqs[:]
    req_list = WORLD.alltoall(req_list)

    sub_data = WORLD.alltoall(sub_data)

    for rank in range(self.size):
      if rank == self.rank:
        continue
      else:
        self.write_local(req_list[rank], sub_data[rank], acc)
    WORLD.barrier()

  def write(self, area, data, acc = 'add'):
    if acc == 'no':
      sub_area = self.local_area & area
      sub_data = data.get().__getitem__(sub_area.offset(area._from).slice)
      self.write_local(sub_area, sub_data)
      return

    reqs = [None] * self.size
    local_subs = [None] * self.size
    if self.unique and area in self.local_area:
      self.write_local(area, data)
    else:
      for rank, a in self.area_dict.iteritems():
        sub_area = a & area
        if sub_area is not None:
          sub_data = data.get().__getitem__(sub_area.offset(area._from).slice)
        else:
          sub_data = None
        if rank == self.rank:
          self.write_local(sub_area, sub_data)
        else:
          reqs[rank] = sub_area
          local_subs[rank] = sub_data
    self.write_remote(reqs, local_subs, acc)

  def merge(self, subs, area):
    subs = {sub_area: sub_array for sub_area, sub_array in subs.iteritems() if sub_array is not None}
    if self.slice_method == DistMethod.Square:
      first, second = self.slice_dim
      row_from = area._from.point[first]
      a = sorted([sub_area for sub_area in subs if sub_area._from.point[first] == row_from], key = lambda x: x._to.point[second])
      rst = np.concatenate(tuple([subs[sub] for sub in a]), axis = second)
      while True:
        row_from = a[0]._to.point[first] + 1
        a = sorted([sub_area for sub_area in subs if sub_area._from.point[first] == row_from], key = lambda x: x._to.point[second])
        if not a: break;
        else:
          tmp = np.concatenate(tuple([subs[sub] for sub in a]), axis = second)
          rst = np.concatenate((rst, tmp), axis = first)
      return rst
    elif self.slice_method == DistMethod.Stripe:
      dim = self.slice_dim
      a = sorted(subs.keys(), key = lambda x : x._from.point[dim])
      rst = np.concatenate(tuple([subs[sub] for sub in a]), axis = dim)
      return rst
    else:
      assert False, 'No implementation'

  @property
  def local_shape(self):
    return self.local_data.shape

  def make_square_area(self):
    first , second = self.slice_dim
    assert first < second < len(self.global_shape), 'Wrong slice_dim'
    local_nrow = self.global_shape[first] / self.nprow
    local_ncol = local_nrow

    first_pos = int(self.rank / self.nprow)
    second_pos = int(self.rank % self.nprow)

    first_from  = first_pos * local_nrow
    first_to = (first_pos + 1) * local_nrow  if self.size - self.rank >= self.nprow else self.global_shape[first]
    second_from = second_pos * local_ncol
    second_to = (second_pos + 1) * local_ncol if (self.rank + 1) % self.nprow != 0  else self.global_shape[second]

    _from = [0] * len(self.global_shape)
    _to = list(self.global_shape)

    _from[first] = int(first_from)
    _from[second] = int(second_from)
    _to[first] = int(first_to)
    _to[second] = int(second_to)
    _to = [x - 1 for x in _to]
    return Area(Point(*_from), Point(*_to))

  def make_stripe_area(self):
    assert self.slice_dim < len(self.global_shape), 'Wrong slice dim'
    nrow = util.divup(self.global_shape[self.slice_dim], self.size)

    pos_from = nrow * self.rank
    pos_to = min( (self.rank+ 1)* nrow , self.global_shape[self.slice_dim])

    _from = [0] * len(self.global_shape)
    _to = list(self.global_shape)
    _from[self.slice_dim] = pos_from
    _to[self.slice_dim] = pos_to
    _to = [x - 1 for x in _to]
    return Area(Point(*_from) , Point(*_to))

  def check_param(self, other):
    return self.slice_method == other.slice_method and self.slice_dim == other.slice_dim and self.unique == other.unique

  def __add__(self, other):
    c = zeros_like(self)
    if isinstance(other, VArray):
      if self.check_param(other):
        c.local_data = self.local_data + other.local_data
        return c
      elif self.unique == False and other.unique == False:
        c.local_data = self.local_data + other.local_data
        return c
      else:
        assert False
    elif np.isscalar(other):
      c.local_data = self.local_data + other
      return c
    else:
      assert False, 'No implementation'



  def __sub__(self, other):
    c = zeros_like(self)
    if isinstance(other, VArray):
      if self.check_param(other) or self.unique == False and other.unique == False:
        c.local_data = self.local_data - other.local_data
      else:
        assert False
    elif np.isscalar(other):
      c.local_data = self.local_data - other
    else:
      assert False, 'No implementation'

    return c

  def __mult__(self, other):
    if np.isscalar(other):
      c = zeros_like(self)
      garray.copy_to(self.local_data * other, c.local_data)
      return c
    else:
      assert self.check_param(other)
      c = zeros_like(self)
      c.local_data   = self.local_data * other_local_data
      return c
  def __eq__(self, other):
    assert self.check_param(other)
    c = zeros_like(self)
    c.local_data = self.local_data == other.local_data

    return c


  def sum(self):
    local_sum = garray.sum(self.local_data)
    if self.unique:
      return local_sum
    else:
      global_sum = WORLD.allreduce(local_sum)
      return global_sum

  def max(self):
    local_max = garray.max(self.local_data)
    if self.unique:
      return local_max
    else:
      global_max = WORLD.allreduce(local_max, op = max)
      return global_max

  def cross_communicate(self, stride, filter_size, padding = 0):
    r, c = self.slice_dim

    half_filter_size = (filter_size - 1) /2
    if stride != 1:
      global_row_begin_centroid = global_col_begin_centroid = half_filter_size - padding

      row_begin_centroid = global_row_begin_centroid
      col_begin_centroid = global_col_begin_centroid

      while row_begin_centroid < self.local_area._from[r]: row_begin_centroid += stride
      while col_begin_centroid < self.local_area._from[c]: col_begin_centroid += stride

      row_end_centroid = row_begin_centroid
      col_end_centroid = col_begin_centroid

      while row_end_centroid < self.local_area._to[r]: row_end_centroid += stride
      row_end_centroid -= stride
      while col_end_centroid < self.local_area._to[c]: col_end_centroid += stride
      col_end_centroid -= stride


      row_up = half_filter_size - (row_begin_centroid - self.local_area._from[r])
      row_down = half_filter_size - (self.local_area._to[r] - row_end_centroid)
      col_left = half_filter_size - (col_begin_centroid - self.local_area._from[c])
      col_right = half_filter_size - (self.local_area._to[c] - col_end_centroid)
    else:
      row_up = row_down = col_left = col_right = half_filter_size

    import copy
    cross_from = copy.deepcopy(self.local_area._from)
    cross_to = copy.deepcopy(self.local_area._to)
    #not most top
    if self.local_area._from[r] != 0:
      cross_from[r] -= row_up
    #not most left
    if self.local_area._from[c] != 0:
      cross_from[c] -= col_left
    #not most down
    if self.local_area._to[r] != self.global_area._to[r]:
      cross_to[r] += row_down
    #not most right
    if self.local_area._to[c] != self.global_area._to[c]:
      cross_to[c] += col_right


    self.tmp_local_area = Area(cross_from, cross_to)
    self.tmp_local_data = self.fetch(self.tmp_local_area)

  def pad(self, padding):
    if padding:
      row, col = self.slice_dim
      u, d, l, r = [padding] * 4
      old_shape = list(self.tmp_local_data.shape)
      old_area = copy.deepcopy(self.tmp_local_area)

      #not most top
      if self.local_area._from[row] != 0:
        u = 0
      else:
        old_shape[row] += padding
        old_area._from[row] += padding
        old_area._to[row] += padding
      #not most left
      if self.local_area._from[col] != 0:
        l = 0
      else:
        old_shape[col] += padding
        old_area._from[col] += padding
        old_area._to[col] += padding
      #not most down
      if self.local_area._to[row] != self.global_area._to[row]:
        d = 0
      else:
        old_shape[row] += padding
      #not most right
      if self.local_area._to[col] != self.global_area._to[col]:
        r = 0
      else:
        old_shape[col] += padding


      if u or d or l or r:
        old_tmp = self.tmp_local_area
        np_tmp = np.zeros(tuple(old_shape), dtype = np.float32)
        slices = old_area.offset(self.tmp_local_area._from).slice
        np_tmp[slices] = self.tmp_local_data
        self.tmp_local_data = np_tmp
    self.tmp_local_data = garray.array(np.require(self.tmp_local_data, dtype = np.float32, requirements = 'C'))

  def unpad(self, data, padding):
    if padding == 0:
      return data
    row, col = self.slice_dim
    u, d, l, r = [padding] * 4
    old_shape = list(data.shape)
    old_area = Area.make_area(data.shape)
    old_area = old_area.move(self.local_area._from)

    #not most top
    if self.local_area._from[row] != 0:
      u = 0
    else:
      old_shape[row] -= padding
      old_area._from[row] += padding
    #not most left
    if self.local_area._from[col] != 0:
      l = 0
    else:
      old_shape[col] -= padding
      old_area._from[col] += padding
    #not most down
    if self.local_area._to[row] != self.global_area._to[row]:
      d = 0
    else:
      old_shape[row] -= padding
      old_area._to[row] -= padding
    #not most right
    if self.local_area._to[col] != self.global_area._to[col]:
      r = 0
    else:
      old_shape[col] -= padding
      old_area._to[col] -= padding


    if u or d or l or r:
      np_tmp = data[old_area.offset(self.local_area._from).slice]
      return np_tmp
    return data

  def add(self, other, dst = None, shape = None, axis = 0):
    if isinstanse(other, VArray):
      data = other.local_data
    else:
      data = other

    if axis == len(self.local_shape) - 1:
      tmp = garray.reshape_last(self.local_data) + data
    elif axis == 0:
      tmp = garray.reshape_first(self.local_data) + data
    else:
      assert False, 'No implementation for axis = %s' % axis

    if dst is not None:
      assert self.check_param(dst)
      c = dst
    else:
      c = zeros_like(self)
    c.local_data = tmp.reshape(self.local_shape)
    return c

  def sumto(self, shape = None, axis = 0):
    assert 0 <= axis < len(self.local_shape)
    if axis == 0:
      c = garray.sum(garray.reshape_first(self.local_data), axis = 1)
      if self.unique:
        if (np.isscalar(self.slice_dim) and axis != self.slice_dim) or (axis not in input.slice_dim):
          c = WORLD.allreduce(c)
        else:
          assert False
      return VArray(c, unquie = False)
    elif axis == len(self.local_shape) -1:
      c = garray.sum(garray.reshape_last(self.local_data), axis = 0)
      if self.unique:
        if (np.isscalar(self.slice_dim) and axis != self.slice_dim) or (axis not in self.slice_dim):
          c = WORLD.allreduce(c)
        else:
          assert False
      return VArray(c, unique = False)
    else:
      assert False

  def maxto(self, shape = None, axis = 0):
    assert 0< axis < len(self.local_shape)
    if axis == 0:
      c = garray.max(garray.reshape_first(self.local_data), axis = 1)
      if self.unique:
        if (np.isscalar(self.slice_dim) and axis != self.slice_dim) or (axis not in self.slice_dim):
          c = WORLD.allreduce(c)
        else:
          assert False
      return VArray(c, unquie = False)
    elif axis == len(self.local_shape) -1:
      c = garray.max(garray.reshape_last(self.local_data), axis = 0)
      if self.unique:
        if (np.isscalar(self.slice_dim) and axis != self.slice_dim) or (axis not in self.slice_dim):
          c = WORLD.allreduce(c, op = max)
        else:
          assert False
      return VArray(c, unique = False)
    else:
      assert False

  def argmaxto(self, shape = None, axis = 0):
    assert 0< axis < len(self.local_shape)
    assert self.unique == False, len(self.local_shape) == 2
    c = garray.argmax(self.local_data, axis = 1- axis)
    return VArray(c, unique = False)


  def fill(self, scalar):
    self.local_data.fill(scalar)


def array(a, unique = True, slice_method = DistMethod.Square, slice_dim = (1, 2)):
  return VArray(a, unique, slice_method, slice_dim)

def square_array(a, slice_dim, unique = True):
  return VArray(a, unique, slice_method = DistMethod.Square, slice_dim = slice_dim)

def zeros(shape, dtype = np.float32, unique = True, slice_method = DistMethod.Square, slice_dim = (1, 2)):
  a = np.zeros(shape).astype(np.float32)
  return VArray(a, unique, slice_method, slice_dim)

def zeros_like(like):
  assert isinstance(like, VArray)
  a = np.zeros(like.global_shape).astype(like.dtype)
  return array(a, unique = like.unique, slice_method = like.slice_method, slice_dim = like.slice_dim)
