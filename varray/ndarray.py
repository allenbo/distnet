import pyximport
pyximport.install()
from mpi4py import MPI
from varray.area import Area, Point
import util
from util import issquare
import numpy as np
import math
import copy
import garray
import time
from pycuda import driver

WORLD = MPI.COMM_WORLD
MASTER = 0
size = WORLD.Get_size()
rank = WORLD.Get_rank()

send_data_size = 0
recv_data_size = 0


class DistMethod(object):
  Square = 'square'
  Stripe = 'stripe'


def barrier():
  WORLD.Barrier()

def tobuffer(gpuarray):
  dtype = np.dtype(gpuarray.dtype)
  return garray.make_buffer(gpuarray.ptr, gpuarray.size * dtype.itemsize)



class VArray(object):
  '''A VArray is used to do distributed array based communication on GPU.

  With slice_method and slice_dim, GPU should know the distribute state
  of VArray, and with area, GPU should know what to send out or receive.
  '''

  def __init__(self, array = None, unique = True,
                            slice_method = DistMethod.Square,
                            slice_dim = None, shape = None, local = False):
    '''Have to provider array or shape.

      Array could be a numpy ndarray or GPUArray. The parameter local indicates the array is
      global array or local array. When local is true, the array is local data.

      The given shape has to be the global shape.

      When shape is given, the global shape is determined it, even if the array is given.
      '''
    start = time.time()
    self.rank = rank
    self.world_size = size

    self.unique = unique
    self.slice_method = slice_method
    self.slice_dim = slice_dim

    self.area_dict = {}

    if shape is not None:
      self.global_shape = shape
    elif hasattr(array, 'shape'):
      self.global_shape = array.shape
    else:
      assert False, 'Array has no shape attr'

    if hasattr(array, 'dtype'):
      self.dtype = array.dtype
    elif shape is not None:
      self.dtype = np.float32
    else:
      assert False, 'Array has no dtype attr'

    if not self.unique:
      if array is None and shape is not None:
        array = garray.GPUArray(shape, self.dtype)
      if isinstance(array, garray.GPUArray):
        self.local_data = array
      else:
        self.local_data = garray.array(array)
      self.local_area = Area.make_area(self.local_shape)
      for i in range(self.world_size):
        self.area_dict[i] = self.local_area
    else:
      if self.slice_method is None:
        self.slice_method = DistMethod.Square

      if self.slice_method == DistMethod.Square:
        assert issquare(self.world_size), 'The size of MPI processes has to square'
        assert slice_dim, 'Must specify slice_diim'
        assert len(slice_dim) == 2, 'Length of slice_dim must be 2'

        self.nprow = math.sqrt(self.world_size)

        self.local_area = self.make_square_area(self.rank)
        if array is not None:
          if isinstance(array, garray.GPUArray):
            if not local:
              self.local_data = array.__getitem__(self.local_area.slice)
            else:
              self.local_data = array
          else:
            self.local_data = garray.array(array.__getitem__(self.local_area.slice).copy())
        else:
          assert shape is not None
          self.local_data = garray.GPUArray(self.local_area.shape, dtype = self.dtype)

      elif self.slice_method == DistMethod.Stripe:
        if self.slice_dim is None:
          self.slice_dim = 0
        else:
          assert np.isscalar(self.slice_dim), 'SLice dim has to the a scalar'

        self.local_area = self.make_stripe_area(self.rank)
        if array is not None:
          if isinstance(array, garray.GPUArray):
            if not local:
              self.local_data = array.__getitem__(self.local_area.slice)
            else:
              self.local_data = array
          else:
            if not local:
              self.local_data = garray.array(array.__getitem__(self.local_area.slice).copy())
            else:
              self.local_data = garray.array(array)
        else:
          assert shape is not None
          self.local_data = garray.GPUArray(self.local_area.shape, dtype =self.dtype)
      else:
        assert False, 'No implementation'

      self.area_dict[self.rank] = self.local_area
      self.infer_area_dict()

    self.fetch_recv_cache = {}
    self.fetch_sent_cache = {}
    self.write_recv_cache = {}
    self.write_sent_cache = {}


  def infer_area_dict(self):
    for i in range(self.world_size):
      if self.slice_method == DistMethod.Square:
        self.area_dict[i] = self.make_square_area(i)
      elif self.slice_method == DistMethod.Stripe:
        self.area_dict[i] = self.make_stripe_area(i)

  def sync_area_dict(self):
    barrier()
    rev = WORLD.allgather(self.area_dict[self.rank])
    for i in range(self.world_size):
      self.area_dict[i] = rev[i]

  @property
  def shape(self):
    return self.global_shape


  def copy_from_global(self, input):
    tmp = input.__getitem__(self.local_area.slice)
    assert tmp.shape == self.local_shape, str(tmp.shape) + ' ' + str(self.local_shape)
    self.local_data = tmp

  @property
  def global_area(self):
    return Area.make_area(self.global_shape)

  def gather(self):
    if not self.unique:
      return

    self.unique = False
    global_area = self.global_area

    self.local_data = self.fetch(global_area)
    self.local_area = global_area

    for i in range(self.world_size):
      self.area_dict[i] = self.local_area

  def fetch_local(self, area):
    if area is None:
      return None
    area = area.offset(self.local_area._from)
    if area.id not in self.fetch_sent_cache:
      data = garray.GPUArray(area.shape, dtype = np.float32)
      self.fetch_sent_cache[area.id] = data
    else:
      data = self.fetch_sent_cache[area.id]
    garray.stride_copy(self.local_data, data, area.slice)
    return data

  def fetch_remote(self, reqs):
    global send_data_size
    global recv_data_size
    subs = {}
    recv_data = []
    req_list = reqs[:]
    for req in req_list:
      if req is None:
        recv_data.append(None)
      else:
        if req.id not in self.fetch_recv_cache:
          self.fetch_recv_cache[req.id] = garray.GPUArray(req.shape, dtype = np.float32)
        recv_data.append(self.fetch_recv_cache[req.id])
    req_list = WORLD.alltoall(req_list)

    send_data = [self.fetch_local(req_list[rank]) for rank in range(self.world_size)]
    send_data_size += sum([int(np.prod(x.shape)) * 4 for x in send_data if x is not None])
    send_req = []
    recv_req = []
    for i,data in enumerate(send_data):
      if i == self.rank or data is None:continue
      send_req.append(WORLD.Isend(tobuffer(data), dest = i))

    for i, data in enumerate(recv_data):
      if i == self.rank or data is None:continue
      recv_req.append(WORLD.Irecv(tobuffer(data), source = i))

    for req in send_req: req.wait()
    for req in recv_req: req.wait()
    recv_data_size += sum([int(np.prod(x.shape)) * 4 for x in recv_data if x is not None])

    subs = { reqs[rank]: recv_data[rank] for rank in range(self.world_size)}
    return subs

  def fetch(self, area, padding = 0):
    start = time.time()
    barrier()
    subs = {}
    reqs = [None] * self.world_size
    if area in self.local_area:
      subs[area] = self.fetch_local(area)
      for i in range(self.world_size):
        if i != self.rank:
          reqs[i] = None
    else:
      for rank, a in self.area_dict.iteritems():
        sub_area = a & area
        if rank == self.rank:
          sub_array = self.fetch_local(sub_area)
          subs[sub_area] = sub_array
          reqs[rank] = None
        else:
          reqs[rank] = sub_area
    subs.update(self.fetch_remote(reqs))
    rst = self.merge(subs, area, padding)
    return rst


  def write_local(self, area,  data, acc = 'overwrite'):
    if area is None:
      return
    area = area.offset(self.local_area._from)
    gpu_data = self.local_data
    if acc == 'overwrite':
      gpu_data.__setitem__(area.slice, data)
    elif acc == 'add':
      garray.setitem_sum(gpu_data, area.slice, data)
    else:
      assert False



  def communicate_remote(self, sub_data, recv_data):
    global send_data_size
    global recv_data_size
    send_req = []
    recv_req = []
    send_data_size += sum([int(np.prod(x.shape)) * 4 for x in sub_data if x is not None])
    for i,data in enumerate(sub_data):
      if i == self.rank or data is None:continue
      send_req.append(WORLD.Isend(tobuffer(data), dest = i))

    for i, data in enumerate(recv_data):
      if i == self.rank or data is None:continue
      recv_req.append(WORLD.Irecv(tobuffer(data), source = i))

    for req in send_req: req.wait()
    for req in recv_req: req.wait()
    recv_data_size += sum([int(np.prod(x.shape)) * 4 for x in recv_data if x is not None])



  def write_remote(self, reqs, sub_data, acc):
    recv_data = []
    req_list = reqs[:]
    req_list = WORLD.alltoall(req_list)
    size = 0
    for req in req_list:
      if req is None:
        recv_data.append(None)
      else:
        if req.id not in self.write_recv_cache:
          self.write_recv_cache[req.id] = garray.GPUArray(req.shape, dtype = np.float32)
        recv_data.append(self.write_recv_cache[req.id])
        size += req.size
    self.communicate_remote(sub_data, recv_data)

    for rank in range(self.world_size):
      if rank == self.rank or req_list[rank] is None: continue
      else:
        self.write_local(req_list[rank], recv_data[rank], acc)

  def write(self, area, data, acc = 'add'):
    barrier()
    start = time.time()
    if acc == 'no':
      sub_area = self.local_area & area
      sub_data = data.__getitem__(sub_area.offset(area._from).slice)
      self.write_local(sub_area, sub_data)
      return

    reqs = [None] * self.world_size
    local_subs = [None] * self.world_size
    if self.unique and area in self.local_area:
      self.write_local(area, data)
    else:
      for rank, a in self.area_dict.iteritems():
        sub_area = a & area
        if sub_area is not None:
          offset_area = sub_area.offset(area._from)
          if offset_area.id not in self.write_sent_cache:
            sub_data = garray.GPUArray(offset_area.shape, dtype = np.float32)
            self.write_sent_cache[offset_area.id] = sub_data
          else:
            sub_data = self.write_sent_cache[offset_area.id]
          garray.stride_copy(data, sub_data, offset_area.slice)
        else:
          sub_data = None
        if rank == self.rank:
          self.write_local(sub_area, sub_data)
          reqs[rank] = None
        else:
          reqs[rank] = sub_area
          local_subs[rank] = sub_data
    self.write_remote(reqs, local_subs, acc)

  def merge(self, subs, area, padding = 0):
    subs = {sub_area: sub_array for sub_area, sub_array in subs.iteritems() if sub_array is not None}
    if padding == 0:
      if len(subs) == 1:
        return subs.values()[0]
      min_from = Area.min_from(subs.keys())
      if area.id not in self.fetch_sent_cache:
        free, total = driver.mem_get_info()
        MB = 1024 * 1024
        mem_size = int(np.prod(area.shape) * 4)
        assert mem_size < free, str(free / MB) + str(mem_size/ MB)
        rst = garray.GPUArray(area.shape, dtype = np.float32)
        self.fetch_sent_cache[area.id] = rst
      else:
        rst = self.fetch_sent_cache[area.id]
      stride_func = garray.stride_write
      for sub_area, sub_array in subs.iteritems():
        garray.stride_write(sub_array, rst, sub_area.offset(min_from).slice)
      return rst
    else:
      def get_new_min_from(min_from, slices):
        for i, s in enumerate(slices):
          start = s.start
          if start != 0:
            min_from[i] -= start

        return min_from

      assert padding < 0
      padding = -padding
      new_shape, slices = self.get_pad_info(padding, area.shape, area)
      min_from = Area.min_from(subs.keys())
      area = Area.make_area(new_shape)
      if area.id not in self.fetch_sent_cache:
        rst = garray.zeros(new_shape, dtype = np.float32)
        self.fetch_sent_cache[area.id] = rst
      else:
        rst = self.fetch_sent_cache[area.id]

      if len(subs) == 1:
        garray.stride_write(subs.values()[0], rst, slices)
        return rst
      min_from = get_new_min_from(min_from, slices)

      for sub_area, sub_array in subs.iteritems():
        garray.stride_write(sub_array, rst, sub_area.offset(min_from).slice)
      return rst

  @property
  def local_shape(self):
    return self.local_data.shape

  @property
  def shape(self):
    return self.global_shape

  def make_square_area(self, rank):
    first , second = self.slice_dim
    assert first < second < len(self.global_shape), 'Wrong slice_dim ' + str(len(self.global_shape))
    local_nrow = self.global_shape[first] / self.nprow
    local_ncol = local_nrow

    first_pos = int(rank / self.nprow)
    second_pos = int(rank % self.nprow)

    first_from  = first_pos * local_nrow
    first_to = (first_pos + 1) * local_nrow  if self.world_size - rank >= self.nprow else self.global_shape[first]
    second_from = second_pos * local_ncol
    second_to = (second_pos + 1) * local_ncol if (rank + 1) % self.nprow != 0  else self.global_shape[second]

    _from = [0] * len(self.global_shape)
    _to = list(self.global_shape)

    _from[first] = int(first_from)
    _from[second] = int(second_from)
    _to[first] = int(first_to)
    _to[second] = int(second_to)
    _to = [x - 1 for x in _to]
    return Area(Point(*_from), Point(*_to))

  def make_stripe_area(self, rank):
    assert self.slice_dim < len(self.global_shape), 'Wrong slice dim'
    #nrow = util.divup(self.global_shape[self.slice_dim], self.world_size)
    nrow = self.global_shape[self.slice_dim] / self.world_size

    pos_from = nrow * rank
    pos_to = (rank+ 1)* nrow
    if rank == self.world_size -1 :
      pos_to = self.global_shape[self.slice_dim]

    _from = [0] * len(self.global_shape)
    _to = list(self.global_shape)
    _from[self.slice_dim] = pos_from
    _to[self.slice_dim] = pos_to
    _to = [x - 1 for x in _to]
    return Area(Point(*_from) , Point(*_to))

  def check_param(self, other):
    return self.slice_method == other.slice_method and self.slice_dim == other.slice_dim and self.unique == other.unique

  def __add__(self, other):
    c = allocate_like(self)
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
    c = allocate_like(self)
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

  def __mul__(self, other):
    if np.isscalar(other):
      c = allocate_like(self)
      garray.copy_to(self.local_data * other, c.local_data)
      return c
    else:
      c = allocate_like(self)
      c.local_data   = self.local_data * other.local_data
      return c

  def __div__(self, other):
    if np.isscalar(other):
      c = allocate_like(self)
      garray.copy_to(self.local_data / other, c.local_data)
      return c
    else:
      c = allocate_like(self)
      c.local_data = self.local_data / other.local_data
      return c

  def __eq__(self, other):
    assert self.check_param(other)
    c = allocate_like(self)
    c.local_data = self.local_data == other.local_data

    return c


  def sum(self):
    barrier()
    local_sum = garray.sum(self.local_data)
    if not self.unique:
      return local_sum
    else:
      global_sum = WORLD.allreduce(local_sum)
      return global_sum

  def max(self):
    barrier()
    local_max = garray.max(self.local_data)
    if not self.unique:
      return local_max
    else:
      global_max = WORLD.allreduce(local_max, op = max)
      return global_max

  def cross_communicate(self, stride, filter_size, padding = 0, num_output = None):
    ''' When cross communicate is being called, FastNet is distribued the image cross height and width'''
    assert padding <= 0, str(padding)
    #r, c = self.slice_dim
    #The dimension of height and width
    r, c =  1, 2

    half_filter_size = (filter_size - 1) /2
    if stride != 1:
      global_row_begin_centroid = global_col_begin_centroid = half_filter_size + padding

      row_begin_centroid = global_row_begin_centroid
      col_begin_centroid = global_col_begin_centroid

      while row_begin_centroid < self.local_area._from[r]: row_begin_centroid += stride
      while col_begin_centroid < self.local_area._from[c]: col_begin_centroid += stride

      row_end_centroid = row_begin_centroid
      col_end_centroid = col_begin_centroid

      while row_end_centroid < self.local_area._to[r]: row_end_centroid += stride
      if row_end_centroid != self.local_area._to[r]:
        row_end_centroid -= stride
      while col_end_centroid < self.local_area._to[c]: col_end_centroid += stride
      if col_end_centroid != self.local_area._to[c]:
        col_end_centroid -= stride

      if num_output is not None:
        num_row , num_col = num_output
        diff = num_row - ((row_end_centroid - row_begin_centroid) / stride  + 1)
        if diff != 0:
          # change the centriod, asssume there are 4 GPU
          if self.local_area._from[r] == 0:
            row_end_centroid += diff * stride
          else:
            row_begin_centroid +=  diff * stride
        diff = num_col - ((col_end_centroid - col_begin_centroid) / stride  + 1)
        if diff != 0:
          # change the centriod, asssume there are 4 GPU
          if self.local_area._from[c] == 0:
            col_end_centroid += diff * stride
          else:
            col_begin_centroid +=  diff * stride

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
    self.tmp_local_data = self.fetch(self.tmp_local_area, padding = padding)
    #self.tmp_local_data = self.fetch(self.tmp_local_area)

  def get_pad_info(self, padding, old_shape, old_area):
    #row, col = self.slice_dim
    row, col = 1, 2
    new_shape = list(old_shape)
    new_area = copy.deepcopy(old_area)

    #most top
    if self.local_area._from[row] == 0:
      new_shape[row] += padding
      new_area._from[row] += padding
      new_area._to[row] += padding
    #most left
    if self.local_area._from[col] == 0:
      new_shape[col] += padding
      new_area._from[col] += padding
      new_area._to[col] += padding

    #most down
    if self.local_area._to[row] == self.global_area._to[row]:
      new_shape[row] += padding
    #most right
    if self.local_area._to[col] == self.global_area._to[col]:
      new_shape[col] += padding

    return tuple(new_shape), new_area.offset(old_area._from).slice

  def pad(self, padding):
    assert padding <= 0
    padding = -padding
    new_shape, slices = self.get_pad_info(padding, self.tmp_local_data.shape, self.tmp_local_area)

    if new_shape != self.tmp_local_data.shape:
      tmp = garray.zeros(new_shape, dtype = np.float32)
      tmp[slices] = self.tmp_local_data
      self.tmp_local_data = tmp

  def unpad(self, data, padding):
    if padding == 0:
      return data
    assert padding <= 0
    padding = -padding
    #row, col = self.slice_dim
    row, col = 1, 2
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
      data = data[old_area.offset(self.local_area._from).slice]
    return data

  def add(self, other, dst = None, shape = None, axis = 0):
    if isinstance(other, VArray):
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
      c = allocate_like(self)
    c.local_data = tmp.reshape(self.local_shape)
    return c

  def sumto(self, shape = None, axis = 0):
    barrier()
    assert 0 <= axis < len(self.local_shape)
    if axis == 0:
      c = garray.sum(garray.reshape_first(self.local_data), axis = 1)
      if self.unique:
        if (np.isscalar(self.slice_dim) and axis != self.slice_dim) or (axis not in self.slice_dim):
          c = WORLD.allreduce(c)
        else:
          assert False
      return VArray(c, unique = False)
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
    barrier()
    assert 0< axis < len(self.local_shape)
    if axis == 0:
      c = garray.max(garray.reshape_first(self.local_data), axis = 1)
      if self.unique:
        if (np.isscalar(self.slice_dim) and axis != self.slice_dim) or (axis not in self.slice_dim):
          c = WORLD.allreduce(c)
        else:
          assert False
      return VArray(c, unique = False)
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

  def get(self):
    if not self.unique:
      return self.local_data.get()
    assert False

  def __getitem__(self, key):
    if not self.unique:
      local_data = self.local_data.__getitem__(key)
      c = VArray(local_data, unique = False)
      return c
    assert False

  @property
  def size(self):
    assert not self.unique
    return self.local_data.size

  def reshape(self, shape):
    assert not self.unique
    data = self.local_data
    return VArray(data.reshape(shape), unique = False)


  def mem_free(self):
    self.local_data.gpudata.free()
    for key, value in self.fetch_recv_cache.iteritems():
      value.free()
    for key, value in self.fetch_sent_cache.iteritems():
      value.free()
    for key, value in self.write_recv_cache.iteritems():
      value.free()
    for key, value in self.write_sent_cache.iteritems():
      value.free()



def array(a, dtype = np.float32,unique = True, slice_method = DistMethod.Square, slice_dim = (1, 2)):
  return VArray(a, unique, slice_method, slice_dim)

def square_array(a, slice_dim, unique = True):
  return VArray(a, unique, slice_method = DistMethod.Square, slice_dim = slice_dim)

def zeros(shape, dtype = np.float32, unique = True, slice_method = DistMethod.Square, slice_dim = (1, 2)):
  a = np.zeros(shape).astype(np.float32)
  return VArray(a, unique, slice_method, slice_dim)


def allocate(shape, dtype = np.float32, unique = True, slice_method = DistMethod.Square, slice_dim = (1, 2)):
  if not issquare(size):
    slice_method = DistMethod.Stripe
    slice_dim = 1
  return VArray(array = None, unique =  unique, slice_method = slice_method, slice_dim = slice_dim, shape = shape)

def allocate_like(input):
  return VArray(array = None, unique = input.unique, slice_method = input.slice_method, slice_dim = input.slice_dim, shape = input.shape)

def zeros_like(like):
  assert isinstance(like, VArray)
  a = np.zeros(like.global_shape).astype(like.dtype)
  return array(a, unique = like.unique, slice_method = like.slice_method, slice_dim = like.slice_dim)

def from_stripe(data, to = 's'):
  assert isinstance(data, np.ndarray)
  recv = WORLD.allgather(data.shape)
  shape_list = [None] * size
  for i in range(size):
    shape_list[i] = recv[i]
  shape_len = np.array([len(s) for s in shape_list], dtype = np.int32)
  assert any(shape_len - shape_len[0]) == False, 'Shape must have same length'
  shape_last = np.array([x[-1] for x in shape_list])
  shape = tuple(shape_list[0][:-1]  +  (int(np.sum(shape_last)), ))

  rst = VArray(data, slice_method = DistMethod.Stripe, slice_dim = len(shape_list[0]) -1, shape = shape, local = True)
  rst.gather()
  if to == 's':
    if issquare(size):
      rst = VArray(array = rst.local_data, slice_dim = (1, 2))
    else:
      rst = VArray(array = rst.local_data, slice_dim = 1, slice_method = DistMethod.Stripe)
  elif to == 'u':
    rst = VArray(array = rst.local_data, unique = False)
  return rst
