import pyximport
pyximport.install()

from .area import Area, Point

import math
import copy
import time
import sys

from pycuda import driver
from mpi4py import MPI
import numpy as np

import garray
from distbase.util import issquare
from garray import ConvDataLayout, FCDataLayout, FilterLayout, WeightLayout
from distbase.util import deprecated
from context import Context, default_context

WORLD = MPI.COMM_WORLD
MASTER = 0
size = WORLD.Get_size()
rank = WORLD.Get_rank()

class DistMethod(object):
  Square = 'square'
  Stripe = 'stripe'

def barrier(communicator = WORLD):
  communicator.Barrier()

def tobuffer(gpuarray):
  dtype = np.dtype(gpuarray.dtype)
  return garray.make_buffer(gpuarray.ptr, gpuarray.size * dtype.itemsize)

def to_gpu(obj):
  if isinstance(obj, garray.GPUArray):
    return obj
  return garray.array(np.require(obj, requirements='C'))

class VArray(object):
  '''
  A VArray is used to do distributed array based communication on GPU.

  With slice_method and slice_dim, GPU should know the distribute state
  of VArray, and with area, GPU should know what to send out or receive.
  '''

  def __init__(self, array = None,
                     global_slice_dim = None,
                     group_slice_dim = None,
                     shape = None,
                     context = default_context):
    '''
    Have to provider array or shape.

    Array could be a numpy ndarray or GPUArray. The parameter local indicates the array is
    global array or local array. When local is true, the array is local data. Local array only
    suppor stripe distribution.

    The given shape has to be the global shape.

    When shape is given, the global shape is determined, even if the array is given.
    '''
    assert array is not None or shape is not None
    self.context = context
    self.num_group = self.context.num_group
    self.group_id = self.context.group_id
    self.global_comm = self.context.global_comm
    self.group_comm = self.context.group_comm
    self.master_comm = self.context.master_comm

    self.global_rank = self.context.global_rank
    self.group_rank = self.context.group_rank

    self.global_size = self.context.global_size
    self.group_size = self.context.group_size

    # can't decide the source of the parameters
    self.global_slice_dim = global_slice_dim
    self.group_slice_dim = group_slice_dim

    self.global_area_dict = {}
    self.group_area_dict = {}
    self.all_area_dict = {}
    
    # attributes for array
    self.dtype = np.float32
    if shape is not None:
      shape = tuple(shape)
    else:
      shape = array.shape

    self.global_area = Area.make_area(shape)

    # global attributes
    if self.num_group == 1:
      self.group_area = self.global_area
      if array is None:
        group_array = garray.GPUArray(tuple(shape), self.dtype)
      else:
        group_array = array
      self.global_area_dict[0] = self.group_area
      self.global_slice_dim = None
    else:
      if not self.global_unique:
        # group data
        if array is None:
          group_array = garray.GPUArray(tuple(shape), self.dtype)
        else:
          group_array = array
        # group shape and area
        self.group_area = Area.make_area(group_array.shape)
        # group area dict
        for i in range(self.num_group):
          self.global_area_dict[i] = self.group_area
      else:
        # figure out the group area
        assert np.isscalar(self.global_slice_dim)
        self.infer_area_dict(num_worker = self.num_group,
                             slice_dim = self.global_slice_dim,
                             global_area = self.global_area,
                             area_dict = self.global_area_dict)

        self.group_area = self.global_area_dict[self.group_id]

        # load up group_data
        if array is not None:
          group_array = to_gpu(array[self.group_area.slice])
        else:
          group_array = garray.GPUArray(self.group_shape, dtype = self.dtype)

    assert group_array.shape == self.group_area.shape

    #  group attributes
    if not self.group_unique:
      self.local_area = self.group_area
      self.local_data = to_gpu(group_array)

      for i in range(self.group_size):
        self.group_area_dict[i] = self.local_area

      for i in range(self.global_size):
        group_id = self.context.get_group_id(i)
        group_area = self.global_area_dict[group_id]
        self.all_area_dict[i] = group_area
    else:
      self.infer_area_dict(num_worker = self.group_size,
                           slice_dim = self.group_slice_dim,
                           global_area = self.group_area,
                           area_dict = self.group_area_dict)

      self.local_area = self.group_area_dict[self.group_rank]
      self.local_data = to_gpu(group_array[self.local_area.offset(self.group_area._from).slice])
      
      if self.num_group == 1:
        self.all_area_dict = self.group_area_dict
      else:
        offset = 0
        for i in range(self.num_group):
          group_area = self.global_area_dict[i]
          tmp_area_dict = self.infer_area_dict(num_worker = self.group_size,
                               slice_dim = self.group_slice_dim,
                               global_area = group_area,
                               area_dict = None)
          for j in range(self.group_size):
            self.all_area_dict[offset + j] = tmp_area_dict[j]

          offset += self.group_size

    assert self.local_area.shape == self.local_data.shape

    self.fetch_recv_cache = {}
    self.fetch_sent_cache = {}
    self.write_recv_cache = {}
    self.write_sent_cache = {}

  def infer_area_dict(self, num_worker, slice_dim, global_area, area_dict = None):
    if area_dict is None:
      area_dict = {}
    assert len(area_dict) == 0
    for i in range(num_worker):
      if not np.isscalar(slice_dim):
        area_dict[i] = VArray.make_square_area(i, slice_dim, global_area, num_worker)
      else:
        area_dict[i] = VArray.make_stripe_area(i, slice_dim, global_area, num_worker)
    return area_dict

  @staticmethod
  def make_square_area(rank, slice_dim, global_area, num_worker):
    first, second = slice_dim
    first_start = global_area._from[first]
    second_start = global_area._from[second]
    global_shape = global_area.shape
    assert first < second < len(global_shape), 'Wrong slice_dim ' + str(len(global_shape))
    nprow = int(math.sqrt(num_worker))
    local_nrow = 1.0 * global_shape[first] / nprow
    local_ncol = 1.0 * global_shape[second] / nprow

    first_pos = int(rank / nprow)
    second_pos = int(rank % nprow)

    first_from  = first_pos * local_nrow
    first_to = (first_pos + 1) * local_nrow  if num_worker - rank > nprow else global_shape[first]
    second_from = second_pos * local_ncol
    second_to = (second_pos + 1) * local_ncol if (rank + 1) % nprow != 0  else global_shape[second]

    _from = global_area._from.point[:]
    _to = [x + 1 for x in global_area._to.point]

    _from[first] = int(first_from) + first_start
    _from[second] = int(second_from) + second_start
    _to[first] = int(first_to) + first_start
    _to[second] = int(second_to) + second_start
    _to = [x - 1 for x in _to]
    return Area(Point(*_from), Point(*_to))

  @staticmethod
  def make_stripe_area(rank, slice_dim, global_area, num_worker):
    global_shape = global_area.shape
    assert slice_dim < len(global_shape), 'Wrong slice dim'
    assert 0 <= rank < num_worker, 'Wrong rank %d in %d workers' % (rank, num_worker)
    pos_start = global_area._from[slice_dim]
    nrow = 1.0 * global_shape[slice_dim] / num_worker

    pos_from = int(nrow * rank)
    pos_to = int((rank+ 1)* nrow)
    if rank == num_worker -1:
      pos_to = global_shape[slice_dim]

    _from = global_area._from.point[:]
    _to = [x + 1 for x in global_area._to.point]

    _from[slice_dim] = pos_from + pos_start
    _to[slice_dim] = pos_to + pos_start
    _to = [x - 1 for x in _to]
    return Area(Point(*_from) , Point(*_to))

  @property
  def shape(self):
    return self.global_shape

  @property
  def size(self):
    return np.prod(self.global_shape)

  @property
  def global_unique(self):
    return self.global_slice_dim is not None

  @property
  def group_unique(self):
    return self.group_slice_dim is not None

  @property
  def global_shape(self):
    return self.global_area.shape

  @property
  def group_shape(self):
    return self.group_area.shape

  @property
  def local_shape(self):
    return self.local_area.shape

  def copy_from_global(self, input):
    tmp = input.__getitem__(self.local_area.slice)
    assert tmp.shape == self.local_shape, str(tmp.shape) + ' ' + str(self.local_shape)
    self.local_data = tmp

  def group_gather(self):
    if not self.group_unique:
      return
    self.group_slice_dim = None
    self.local_data = self.group_fetch(self.group_area)
    for i in range(self.group_size):
      self.group_area_dict[i] = self.group_area
    self.local_area = self.group_area
    self.local_shape = self.local_area.shape

  def gather(self):
    if not self.global_unique:
      if not self.group_unique:
        return
      else:
        self.group_gather()
    else:
      self.global_slice_dim = None
      self.group_slice_dim = None

      self.group_area = self.global_area
      self.local_area = self.group_area
      self.local_shape = self.local_area.shape
      self.local_data = sel.fetch(self.global_area)
      for i in range(self.num_group):
        self.global_area_dict[i] = self.global_area
      for i in range(self.group_size):
        self.group_area_dict[i] = self.global_area

  def fetch_local(self, area):
    if area is None:
      return None
    if area == self.local_area:
      return self.local_data
    area = area.offset(self.local_area._from)
    if area.id not in self.fetch_sent_cache:
      data = garray.GPUArray(area.shape, dtype = np.float32)
      self.fetch_sent_cache[area.id] = data
    else:
      data = self.fetch_sent_cache[area.id]
    #data = garray.zeros(area.shape, dtype = np.float32)
    garray.stride_copy(self.local_data, data, area.slice)
    return data

  def _communicate(self, send_data, recv_data, communicator):
    send_req = []
    recv_req = []
    for i, data in enumerate(send_data):
      if data is None:continue
      send_req.append(communicator.Isend(tobuffer(data), dest = i))

    for i, data in enumerate(recv_data):
      if data is None:continue
      recv_req.append(communicator.Irecv(tobuffer(data), source = i))

    for req in send_req: req.wait()
    for req in recv_req: req.wait()

  def fetch_remote(self, reqs, communicator, self_id):
    subs = {}
    req_list = reqs[:]
    num_worker = len(req_list)

    # prepare recv_data
    recv_data = [None] * num_worker
    for i, req in enumerate(req_list):
      if req is not None:
        if req.id not in self.fetch_recv_cache:
          self.fetch_recv_cache[req.id] = garray.GPUArray(req.shape, dtype = np.float32)
        buffer =  self.fetch_recv_cache[req.id]
        #buffer = garray.zeros(req.shape, dtype = np.float32)
        recv_data[i] = buffer

    # preparea send_data
    req_list = communicator.alltoall(req_list)
    send_data = [self.fetch_local(req_list[rank]) for rank in range(num_worker)]
    # communicate with other workers
    self._communicate(send_data, recv_data, communicator)

    subs = {reqs[rank]: recv_data[rank] for rank in range(num_worker)}
    return subs

  def _fetch(self, area, padding, slice_dim, self_id, num_worker, area_dict, communicator):
    barrier(communicator)
    subs = {}
    reqs = [None] * num_worker
    if area in self.local_area:
      subs[area] = self.fetch_local(area)
    else:
      for rank, a in area_dict.iteritems():
        sub_area = a & area
        if rank == self_id:
          sub_array = self.fetch_local(sub_area)
          subs[sub_area] = sub_array
          reqs[rank] = None
        else:
          reqs[rank] = sub_area
    subs.update(self.fetch_remote(reqs, communicator, self_id))
    rst = self.merge(subs, area, padding, slice_dim)
    return rst

  def fetch(self, area, padding = 0, slice_dim = None):
    return self._fetch(area = area,
                       padding = padding,
                       slice_dim = slice_dim,
                       self_id = self.global_rank,
                       num_worker = self.global_size,
                       area_dict = self.all_area_dict,
                       communicator = self.global_comm)
  
  def group_fetch(self, area, padding = 0, slice_dim = None):
    return self._fetch(area = area,
                       padding = padding,
                       slice_dim = slice_dim,
                       self_id = self.group_rank,
                       num_worker = self.group_size,
                       area_dict = self.group_area_dict,
                       communicator = self.group_comm)

  def merge(self, subs, area, padding = 0, slice_dim = None):
    subs = {sub_area: sub_array for sub_area, sub_array in subs.iteritems() if sub_array is not None}
    if padding == 0:
      if len(subs) == 1:
        return subs.values()[0]
      min_from = Area.min_from(subs.keys())
      if area.id not in self.fetch_sent_cache:
        rst = garray.GPUArray(area.shape, dtype = np.float32)
        self.fetch_sent_cache[area.id] = rst
      else:
        rst = self.fetch_sent_cache[area.id]
      #rst = garray.zeros(shape = area.shape, dtype = np.float32)
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
      new_shape, slices = self.get_pad_info(padding, area.shape, area, slice_dim)
      min_from = Area.min_from(subs.keys())
      area = Area.make_area(new_shape)
      if area.id not in self.fetch_sent_cache:
        rst = garray.zeros(new_shape, dtype = np.float32)
        self.fetch_sent_cache[area.id] = rst
      else:
        rst = self.fetch_sent_cache[area.id]
      #rst = garray.zeros(new_shape, dtype = np.float32)

      if len(subs) == 1:
        garray.stride_write(subs.values()[0], rst, slices)
        return rst
      min_from = get_new_min_from(min_from, slices)

      for sub_area, sub_array in subs.iteritems():
        garray.stride_write(sub_array, rst, sub_area.offset(min_from).slice)
      return rst

  def write_local(self, area,  data, acc = False):
    if area is None:
      return
    area = area.offset(self.local_area._from)
    gpu_data = self.local_data
    if acc:
      garray.setitem_sum(gpu_data, area.slice, data)
    else:
      if data is gpu_data: return
      gpu_data[area.slice] =  data

  def write_remote(self, reqs, sub_data, communicator):
    req_list = reqs[:]
    num_worker = len(req_list)
    recv_data = [None] * num_worker

    req_list = communicator.alltoall(req_list)
    # prepare recv_data
    for i, req in enumerate(req_list):
      if req is not None:
        recv_data[i] = garray.zeros(req.shape, dtype = np.float32)

    send_data = sub_data
    self._communicate(send_data, recv_data, communicator)

    for rank in range(num_worker):
      #print 'rank:%d , req_list[rank]:%s' % (rank, req_list[rank])
      if req_list[rank] is None: continue
      else:
        #print recv_data[rank][0:1, :, :, 0:1]
        self.write_local(req_list[rank], recv_data[rank], acc = True)
  
  def _partial_write(self, area, data):
    if data is self.local_data: return

    sub_area = self.local_area & area
    if sub_area is None: return

    if sub_area.shape == data.shape:
      sub_data = data
    else:
      sub_data = data.__getitem__(sub_area.offset(area._from).slice)

    self.write_local(sub_area, sub_data)
  
  def _write(self, area, data, propagate, unique, self_id, num_worker, area_dict, communicator):
    barrier(communicator)
    if not propagate:
      self._partial_write(area, data)
      return

    assert area.shape == data.shape
    reqs = [None] * num_worker
    local_subs = [None] * num_worker
    if unique and area in self.local_area:
      self.write_local(area, data)
    else:
      for rank, a in area_dict.iteritems():
        sub_area = a & area

        if sub_area is not None:
          offset_area = sub_area.offset(area._from)
          if offset_area.id not in self.write_sent_cache:
            sub_data = garray.GPUArray(offset_area.shape, dtype = np.float32)
            self.write_sent_cache[offset_area.id] = sub_data
          else:
            sub_data = self.write_sent_cache[offset_area.id]
          #sub_data = garray.GPUArray(offset_area.shape, dtype = np.float32)
          garray.stride_copy(data, sub_data, offset_area.slice)
        else:
          sub_data = None

        if rank == self_id:
          self.write_local(sub_area, sub_data)
          reqs[rank] = None
        else:
          reqs[rank] = sub_area
          local_subs[rank] = sub_data
    self.write_remote(reqs, local_subs, communicator)
    
  def group_write(self, area, data, propagate = True):
    self._write(area = area,
                data = data,
                propagate = propagate,
                unique = self.group_unique,
                num_worker = self.group_size,
                self_id = self.group_rank,
                area_dict = self.group_area_dict,
                communicator = self.group_comm)

  def master_write(self):
    if self.num_group == 1: return
    assert self.global_unique == False and self.group_unique == False
    barrier(WORLD)
    if self.group_rank == 0:
      self._write(area = self.local_area,
                  data = self.local_data,
                  propagate = True,
                  unique = False,
                  self_id = self.group_id,
                  num_worker = self.num_group,
                  area_dict = {i:self.local_area for i in range(self.num_group)},
                  communicator = self.master_comm)

  def group_bcast(self):
    assert self.group_unique == False
    barrier(self.group_comm)
    if self.group_rank != 0:
      recv_data = garray.empty_like(self.local_data)
    else:
      recv_data = self.local_data

    self.group_comm.Bcast(tobuffer(recv_data), root = 0)
    if self.group_rank != 0:
      self.local_data = recv_data

  def group_reduce(self):
    assert self.group_unique == False
    barrier(self.group_comm)
    if self.group_rank == 0:
      recv_buff = []
      requests = []
      for i in range(1, self.group_size):
        buff = garray.empty_like(self.local_data)
        requests.append(self.group_comm.Irecv(tobuffer(buff), source = i))
        recv_buff.append(buff)
      for req in requests: req.wait()
      for buff in recv_buff:
        self.local_data += buff
    else:
      request = self.group_comm.Isend(tobuffer(self.local_data), dest = 0)
      request.wait()

  def write(self, area, data, propagate = True, debug = False):
    if self.global_unique:
      self.group_write(area, data, propagate)
    else:
      if not self.group_unique:
        self._partial_write(area, data)
        if propagate:
          self.group_reduce()
          self.master_write()
          self.group_bcast()
      else:
        self.group_write(area, data, propagate)
    
  def check_param(self, other):
    return self.global_slice_dim == other.global_slice_dim and self.group_slice_dim == other.group_slice_dim

  def __add__(self, other):
    '''
    Now this function is only called at FC layer, when adding bias up to output, and the bias could
    be splitted, so we still need to check the parameters
    '''
    c = allocate_like(self)
    if isinstance(other, VArray):
      if self.check_param(other):
        c.local_data = self.local_data + other.local_data
        return c
      elif self.group_unique == False and other.group_unique == False:
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
      if self.check_param(other) or self.group_unique == False and other.group_unique == False:
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
      c.local_data  = self.local_data * other.local_data
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
    c = allocate_like(self)
    if isinstance(other, garray.GPUArray):
      if self.group_unique == False:  
        assert other.shape == self.local_shape
        c.local_data = self.local_data == other
        return c
      else:
        c.local_data = self.local_shape == other[self.local_area.slice]
      return c
    else:
      assert False

  def sum(self):
    local_sum = garray.sum(self.local_data)
    if not self.group_unique:
      return local_sum
    else:
      global_sum = WORLD.allreduce(local_sum)
      return global_sum

  def group_global_communicate(self):
    self.tmp_local_area = Area.make_area(self.group_shape)
    if self.group_unique:
      self.tmp_local_data = self.group_fetch(self.group_area)
    else:
      self.tmp_local_data = self.local_data

  def global_global_communicate(self):
    self.tmp_local_area = Area.make_area(self.global_shape)
    if self.global_unique:
      self.tmp_local_data = self.fetch(self.tmp_local_area)
    else:
      self.group_global_communicate()

  def channel_communicate(self, rank, slice_dim, padding = 0):
    if padding == 0:
      self.tmp_local_area = Area.make_stripe_area(rank, slice_dim)
    else:
      tmp_area = Area.make_stripe_area(rank, slice_dim)
      if tmp_area._from[slice_dim] != 0:
        tmp_area._from[slice_dim] -= padding
      if tmp_area._to[slice_dim] != self.global_area._to[slice_dim]:
        tmp_area._to[slice_dim] += padding
      self.tmp_local_area = tmp_area
    self.tmp_local_data = self.fetch(self.tmp_local_area)

  def batch_communicate(self, rank, slice_dim):
    self.tmp_local_area = VArray.make_stripe_area(rank, slice_dim, self.group_area, self.group_size)
    self.tmp_local_data = self.group_fetch(self.tmp_local_area)

  def image_communicate(self, slice_dim, stride, filter_size, padding = 0, output_area = None):
    assert padding <= 0, str(padding)
    r, c = slice_dim
    if filter_size != 0:
      half_filter_size = (filter_size - 1) /2
    
      from_point = output_area._from
      to_point = output_area._to

      row_begin_centroid = from_point[r] * stride + padding + half_filter_size
      row_end_centroid = to_point[r] * stride + padding + half_filter_size
      col_begin_centroid = from_point[c] * stride + padding + half_filter_size
      col_end_centroid = to_point[c] * stride + padding + half_filter_size

      row_begin = max(row_begin_centroid - half_filter_size, 0)
      row_end = min(row_end_centroid + half_filter_size, self.group_shape[r] - 1)
      col_begin = max(col_begin_centroid - half_filter_size, 0)
      col_end = min(col_end_centroid + half_filter_size, self.group_shape[c] - 1)
      
      _from = self.group_area._from[:]
      _to = self.group_area._to[:]

      _from[r] = row_begin
      _to[r] = row_end
      _from[c] = col_begin
      _to[c] = col_end
      self.tmp_local_area = Area(Point(*_from), Point(*_to))
    else:
      self.tmp_local_area = copy.deepcopy(output_area)
    self.tmp_local_data = self.group_fetch(self.tmp_local_area, padding = padding, slice_dim = slice_dim)

  def local_patch(self, data):
    # reversed way against communicate
    sub_area = self.local_area & self.tmp_local_are
    sub_data = data.__getitem__(sub_area.offset(area._from).slice)
    self.write_local(sub_area, sub_data)
    
  def get_pad_info(self, padding, old_shape, old_area, slice_dim = None):
    #row, col = self.slice_dim
    row, col = slice_dim
    new_shape = list(old_shape)
    new_area = copy.deepcopy(old_area)

    #most top
    if old_area._from[row] == 0:
      new_shape[row] += padding
      new_area._from[row] += padding
      new_area._to[row] += padding
    #most left
    if old_area._from[col] == 0:
      new_shape[col] += padding
      new_area._from[col] += padding
      new_area._to[col] += padding

    #most down
    if old_area._to[row] == self.group_area._to[row]:
      new_shape[row] += padding
    #most right
    if old_area._to[col] == self.group_area._to[col]:
      new_shape[col] += padding

    return tuple(new_shape), new_area.offset(old_area._from).slice

  def pad(self, padding, slice_dim):
    if padding == 0: return
    assert padding <= 0
    padding = -padding
    new_shape, slices = self.get_pad_info(padding, self.tmp_local_data.shape, self.tmp_local_area, slice_dim)

    if new_shape != self.tmp_local_data.shape:
      tmp = garray.zeros(new_shape, dtype = np.float32)
      garray.stride_write(self.tmp_local_data, tmp, slices)
      self.tmp_local_data = tmp

  def unpad(self, data, padding, old_shape, old_area, slice_dim, debug = False):
    if padding == 0:
      return data
    assert padding <= 0
    padding = -padding
    row, col = slice_dim
    u, d, l, r = [padding] * 4
    new_shape = list(old_shape)
    new_area = copy.deepcopy(old_area)

    #not most top
    if old_area._from[row] != 0:
      u = 0
    else:
      new_shape[row] -= padding
      new_area._from[row] += padding
      new_area._to[row] += padding
    #not most left
    if old_area._from[col] != 0:
      l = 0
    else:
      new_shape[col] -= padding
      new_area._from[col] += padding
      new_area._to[col] += padding
    #not most down
    if old_area._to[row] != self.group_area._to[row]:
      d = 0
    else:
      new_shape[row] -= padding
    #not most right
    if old_area._to[col] != self.group_area._to[col]:
      r = 0
    else:
      new_shape[col] -= padding

    if u or d or l or r:
      data = data[new_area.offset(old_area._from).slice]
    return data
  
  def fill(self, scalar):
    self.local_data.fill(scalar)

  def get(self):
    if not self.global_unique and not self.group_unique:
      return self.local_data.get()
    else:
      return self.fetch(self.global_area).get()

  def __getitem__(self, key):
    if not self.unique:
      local_data = self.local_data.__getitem__(key)
      c = VArray(local_data, unique = False)
      return c
    assert False

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

  def printout(self, name, row_from = 0, row_to = 0, col_from = 0, col_to = 0):
    barrier(self.global_comm)
    if not self.group_unique and not self.global_unique:
      x = self.local_data
    else:
      x = self.fetch(self.global_area)
    
    if self.global_rank == 1:
      x.printout(name, row_from = row_from, row_to = row_to, col_from =  col_from, col_to = col_to)
    #self.local_data.printout(name)
    barrier(self.global_comm)

def array(obj, global_slice_dim, group_slice_dim, context = default_context):
  return VArray(array = obj,
                global_slice_dim = global_slice_dim,
                group_slice_dim = group_slice_dim,
                context = context)

def allocate(shape, global_slice_dim, group_slice_dim, context = default_context):
  return VArray(array = None,
                global_slice_dim = global_slice_dim,
                group_slice_dim = group_slice_dim,
                shape = shape,
                context = context)

def zeros(shape, global_slice_dim, group_slice_dim, context = default_context):
  va = allocate(shape = shape,
                global_slice_dim = global_slice_dim,
                group_slice_dim = group_slice_dim,
                context = context)
  va.fill(0)
  return va

def allocate_like(input):
  return VArray(array = None,
                global_slice_dim = input.global_slice_dim,
                group_slice_dim = input.group_slice_dim,
                shape = input.shape,
                context = input.context)

def zeros_like(like):
  assert isinstance(like, VArray)
  va = allocate_like(like)
  va.fill(0)
  return va
