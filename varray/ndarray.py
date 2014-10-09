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
from distbase import util
from distbase.util import issquare
from distbase.monitor import MONITOR
from garray import ConvDataLayout, FCDataLayout, FilterLayout, WeightLayout
from distbase.util import deprecated
from context import Context, default_context
from cache import Cache

WORLD = MPI.COMM_WORLD
MASTER = 0
size = WORLD.Get_size()
rank = WORLD.Get_rank()

MVAPICH2 = False
INNER = True

def barrier(communicator = WORLD):
  '''
  Sychronize the give communicator world
  @param communcator[WORLD]:Communicator object in MPI
  '''
  communicator.Barrier()

def tobuffer(gpuarray):
  '''
  Create a read-write buffer for GPUArray
  @param gpuarray:GPUArray object
  '''
  dtype = np.dtype(gpuarray.dtype)
  return garray.make_buffer(gpuarray.ptr, gpuarray.size * dtype.itemsize)

def to_gpu(obj):
  '''
  A wrapper for numpy array and GPUArray
  @param obj:numpy.ndarray or pycuda.gpuarray.GPUArray
  '''
  if isinstance(obj, garray.GPUArray):
    return obj
  return garray.array(np.require(obj, requirements='C'))

class VArray(object):
  '''
  A VArray is used to do array based communication across multi GPU.

  A VArray could be divided(or distributed) in multi dimensions.
  Since VArray is designed specifically for Neural Network computation, the distribution has a corresponding
  parallelism.

    - Data parallelism:
      -- Batch: The cutting dimension(slice_dim) is the dimension of batch, which in cudaconv data
         layout, it's last dimension.
      -- Image: The cutting dimension is the dimension of image height and width. Yes, VArray
         support two dim slice at once.

      Both parallelism only distribute VArray in one way, so the number of groups is 1 and
      global_unique is False.

      -- BatchImage: This is the mix of batch and image data parallelism, in which case, VArray will
         be first distributed to several groups in batch, and within each group, VArray will continue
         to be distributed by image.
      -- BatchBatch: First distributed by batch to sevaral groups, then in the group, continue to be
         distributed by batch.

      Bothe parallelism distribute VArray in two ways, (first batch, second batch or image), so the
      number of groups is greater than one and global_unique is True.

      In data parallelism, weight and bias are all shared across multi GPUs, meaning global_unique
      and group_unique area both False

    - Model parallelism:
      Model parallelism has to stick to one-dimension, meaning not like data parallelism, the number
      of groups is always one, so the global_unique is False always.

      Weights and biases in model parallelism are unique across GPUs, so the group_unique is True.

    VArray is built upon MPI. It needs MPI to boostrap itself to figure out where and which part of
    global array lays. When initializing, VArray, would infer the area of any other parts of this
    array and construct an area dictionary. This dictionary is what VArray will query when it needs to
    know which GPU holds which parts of data.

    There are several ways to create a VArray.
    @function array:    Divided numpy ndarray or GPUArray into VArray given global_slice_dim and
                        group_slice_dim, and context when necessary.

    @function allocate: Create a VArray given a global shape and other information, like slice_dim
                        and context.

    @function zeros:    Call allocate first and then fill with 0.

    @function random_uniform: Create a VArray who's data uniformly distribute between 0 and 1.

    @assemble:          Assemble multiply data in different GPUs by wrapping a VArray wrapper for them.

    @function *_like:   Given another VArray, create a similar one.


    When initiliazing an VArray, you have to provide an array or a global shape. global_slice_dim ==
    None means there is only one group in VArray. group_slice_dim == None means VArray doesn't
    divide data within each group.


    @property
    DATA: The actual data on this GPU. It has to be a GPUArray

    @property
    local_area: The area that this GPU is holding

    @property
    shape: global shape of VArray

    @property
    global_shape: same as shape

    @property
    group_shape: the shape of the group which this GPU belongs to

    @property
    local_shape: the shape of the actual data on this GPU


    There area two sets of class member methods in VArray, communication methods and computation
    methods.
  '''

  def __init__(self, array = None,
                     global_slice_dim = None,
                     group_slice_dim = None,
                     shape = None,
                     context = default_context):
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

    self._async_flag = False

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

    self.use_cache = False
    self.cache = Cache()

  def _sync(self):
    for req in self._async_reqs:
      req.wait()
    for data in self._recv_data:
      if data is not None:
        self.local_data += data
    self._async_reqs = None
    self._recv_data = None
    self._async_flag = False

  @property
  def DATA(self):
    if self._async_flag:
      self._sync()
    return self.local_data

  @DATA.setter
  def DATA(self, value):
    self.local_data = value

  def has_local_cache(self):
    '''
    @utility.
    Check if VArray has cache
    '''
    return hasattr(self, 'local_cache')

  def get_gpuarray(self, area, zeroed = False):
    '''
    @utility.intenal
    Get a gpuarry by area
    '''
    if self.use_cache:
      array = self.cache.get(area)
    else:
      array = garray.GPUArray(area.shape, np.float32)

    if zeroed:
      array.fill(0)
    return array

  def infer_area_dict(self, num_worker, slice_dim, global_area, area_dict = None):
    '''
    @utility.internal
    Infer the areas of other GPUs
    '''
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
    '''
    @utility.internal
    Make squre area given slice_dim, area and number of workers
    '''
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
    '''
    @utility.internal
    Make stripe area given slice_dim, area and number of workers
    '''
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

  def regroup_like(self, other):
    '''
    @utility
    Change th group structure to other VArray
    '''
    self.global_slice_dim = other.global_slice_dim
    self.global_area_dic = other.global_area_dict
    self.group_area = other.group_area

  def copy_from_group(self, input):
    '''
    @utility
    @param input:GPUArray, assert input.shape = group_shape
    Take input as group data and copy correspoinding part of input into each GPU.
    '''
    assert input.shape == self.group_area.shape
    self.local_data = input[self.local_area.offset(self.group_area._from).slice]

  def group_gather(self):
    '''
    @communication
    Gather group data and store the complete copy on each GPU, which makes group_unique False
    '''
    if not self.group_unique:
      return

    self.group_slice_dim = None
    self.local_data = self.group_fetch(self.group_area)

    for i in range(self.group_size):
      self.group_area_dict[i] = self.group_area
    self.local_area = self.group_area

  def gather(self):
    '''
    @communication
    Gather global data and store the complete copy on each GPU, which makes both global_unique and
    group_unique False
    '''
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
    '''
    @utility
    @param area:Area object, area has to be contained by local_area
    Get local data at given area
    '''
    if area is None:
      return None
    if area == self.local_area:
      return self.local_data
    area = area.offset(self.local_area._from)
    data = self.get_gpuarray(area)
    garray.stride_copy(self.local_data, data, area.slice)
    return data

  def _communicate(self, send_data, recv_data, communicator, async = False):
    '''
    @communication.internal
    @param send_data:list of gpu data that need to be sent out
    @param recv_data:list of gpu data that will receive data from other GPUs
    @param communicator:MPI world
    @param async:if True, exit function after posting requests w/o actually finish requests, and set
                 async_flag to True
    '''
    _ = time.time()
    send_req = []
    recv_req = []
    for i, data in enumerate(send_data):
      if data is None:continue
      send_req.append(communicator.Isend(tobuffer(data), dest = i))

    for i, data in enumerate(recv_data):
      if data is None:continue
      recv_req.append(communicator.Irecv(tobuffer(data), source = i))

    if async:
      self._async_reqs = send_req + recv_req
      self._recv_data = recv_data
    else:
      for req in send_req + recv_req: req.wait()
    if INNER: MONITOR.add_comm(time.time() - _)

  def fetch_remote(self, reqs, communicator, self_id):
    '''
    @communication.internal
    @param reqs:list of requests for remote peers, elements are area objects
    @param communicator:MPI world
    @param self_id:self rank inside the communicator

    @return dict<k, v>, key is area in the reqs, value is data from remote GPUs.
    '''
    _ = time.time()
    subs = {}
    req_list = reqs[:]
    num_worker = len(req_list)

    # prepare recv_data
    recv_data = [None] * num_worker
    for i, req in enumerate(req_list):
      if req is not None:
        recv_data[i] = self.get_gpuarray(req)

    # preparea send_data
    req_list = communicator.alltoall(req_list)
    send_data = [self.fetch_local(req_list[rank]) for rank in range(num_worker)]
    if INNER: MONITOR.add_marshall(time.time() - _)
    # communicate with other workers
    self._communicate(send_data, recv_data, communicator)

    subs = {reqs[rank]: recv_data[rank] for rank in range(num_worker)}
    return subs

  def _fetch(self, area, padding, slice_dim, self_id, num_worker, area_dict, communicator):
    '''
    @communication.internal
    @param area:the target area, can cross multiply GPUs.
    @param padding:padding of final data
    @param slice_dim:dividing dimension of data pieces from multi GPUs
    @param self_id:self rank inside the communicator world
    @param area_dict:the area dictionay to query
    @param communicator:MPI world

    @return GPUArray data of the area with padding
    '''
    barrier(communicator)
    _ = time.time()
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
    if INNER: MONITOR.add_marshall(time.time() - _)
    subs.update(self.fetch_remote(reqs, communicator, self_id))
    _ = time.time()
    rst = self.merge(subs, area, padding, slice_dim)
    if INNER: MONITOR.add_merge(time.time() - _)
    barrier(communicator)
    return rst

  def fetch(self, area, padding = 0, slice_dim = None):
    '''
    @communication
    @param area:the target area
    @param padding:number of padding for final data
    @param slice_dim:dividing dimension of receving data pieces

    @return GPUArray data of the target area with padding

    This is a global fetch method
    '''
    return self._fetch(area = area,
                       padding = padding,
                       slice_dim = slice_dim,
                       self_id = self.global_rank,
                       num_worker = self.global_size,
                       area_dict = self.all_area_dict,
                       communicator = self.global_comm)

  def group_fetch(self, area, padding = 0, slice_dim = None):
    '''
    @communication
    This is a group fetch method
    '''
    return self._fetch(area = area,
                       padding = padding,
                       slice_dim = slice_dim,
                       self_id = self.group_rank,
                       num_worker = self.group_size,
                       area_dict = self.group_area_dict,
                       communicator = self.group_comm)

  def merge(self, subs, area, padding = 0, slice_dim = None):
    '''
    @utility.internal
    @param subs:dict object, from fetch_remote <k,v> = <area, GPUArray data>
    @param area:the target area
    @param padding:padding
    @param slice_dim:divding dimension of those GPUArray data pieces

    @return  GPUArray data that covered in the area, with padding
    '''
    subs = {sub_area: sub_array for sub_area, sub_array in subs.iteritems() if sub_array is not None}
    if padding == 0:
      if len(subs) == 1:
        return subs.values()[0]
      dst = self.get_gpuarray(area)
      min_from = Area.min_from(subs.keys())
      for sub_area, sub_array in subs.iteritems():
        garray.stride_write(sub_array, dst, sub_area.offset(min_from).slice)
      return dst
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
      dst = self.get_gpuarray(area, zeroed = True)
      if len(subs) == 1:
        garray.stride_write(subs.values()[0], dst, slices)
        return dst
      min_from = get_new_min_from(min_from, slices)

      for sub_area, sub_array in subs.iteritems():
        garray.stride_write(sub_array, dst, sub_area.offset(min_from).slice)
      return dst

  def write_local(self, area,  data, acc = False):
    '''
    @utility.internal
    @param area:destination of the writing operation
    @param data:source data
    @param acc:if True, add to original data, otherwise, over write original
    '''
    if area is None:
      return
    area = area.offset(self.local_area._from)
    gpu_data = self.local_data
    if acc:
      garray.setitem_sum(gpu_data, area.slice, data)
    else:
      if data is gpu_data: return
      gpu_data[area.slice] =  data

  def write_remote(self, reqs, sub_data, communicator, async = False):
    '''
    @communication.internal
    @param reqs:list of remote areas that will be written to
    @param sub_data:list of GPUArray data pieces that will be sent out
    @param communicator:MPI world
    @param async:if True, async write
    '''
    _ = time.time()
    req_list = reqs[:]
    num_worker = len(req_list)
    recv_data = [None] * num_worker

    req_list = communicator.alltoall(req_list)
    # prepare recv_data
    for i, req in enumerate(req_list):
      if req is not None:
        recv_data[i] = self.get_gpuarray(req)

    send_data = sub_data
    if INNER: MONITOR.add_marshall(time.time() - _)
    self._communicate(send_data, recv_data, communicator, async)

    if not async:
      _ = time.time()
      if self.global_unique == False and self.group_unique == False:
        for data in recv_data:
          if data is not None:
            self.local_data += data
      else:
        for rank in range(num_worker):
          if req_list[rank] is None: continue
          else:
            self.write_local(req_list[rank], recv_data[rank], acc = True)
      if INNER: MONITOR.add_merge(time.time() - _)
    barrier(communicator)

  def _partial_write(self, area, data):
    '''
    @utility.internal
    @param area:An area object, which probably cover more than one GPU
    @param data:GPUArray data, which probably will be written to more than one GPU

    Only write into local storage
    '''
    if data is self.local_data: return

    sub_area = self.local_area & area
    if sub_area is None: return

    if sub_area.shape == data.shape:
      sub_data = data
    else:
      sub_data = data.__getitem__(sub_area.offset(area._from).slice)

    self.write_local(sub_area, sub_data)

  def _write(self, area, data, propagate, unique, self_id, num_worker, area_dict, communicator, async = False):
    '''
    @communication.internal
    @param area:target area that will be written
    @param data:source data that will be written to multiply GPUs
    @param progagate:whether send data to remote GPUs
    @param unique:whether data is unique inside the communicator world
    @param self_id:self rank inside the communicator world
    @param num_worker:number of peer GPUs inside the communicator world
    @param area_dict:area dictionary of this world
    @param communicator:MPI world
    @param async:if True, write asynchronousely
    '''
    barrier(communicator)
    if not propagate:
      _ = time.time()
      self._partial_write(area, data)
      if INNER: MONITOR.add_marshall(time.time() - _)
      return

    _ = time.time()
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
          sub_data = self.get_gpuarray(offset_area)
          garray.stride_copy(data, sub_data, offset_area.slice)
        else:
          sub_data = None

        if rank == self_id:
          self.write_local(sub_area, sub_data)
          reqs[rank] = None
        else:
          reqs[rank] = sub_area
          local_subs[rank] = sub_data
    if INNER: MONITOR.add_marshall(time.time() - _)
    self.write_remote(reqs, local_subs, communicator, async)

  def group_write(self, area, data, propagate = True):
    '''
    @communication
    This is a group write function
    '''
    self._write(area = area,
                data = data,
                propagate = propagate,
                unique = self.group_unique,
                num_worker = self.group_size,
                self_id = self.group_rank,
                area_dict = self.group_area_dict,
                communicator = self.group_comm)

  def global_write(self, area, data, propagate = True):
    '''
    @communication
    This is a global write function
    '''
    self._write(area = area,
                data = data,
                propagate = propagate,
                unique = self.global_unique and self.group_unique,
                num_worker = self.global_size,
                self_id = self.global_rank,
                area_dict = self.all_area_dict,
                communicator = self.global_comm)

  def master_write(self):
    '''
    @communication
    This communication world consists of group masters only
    '''
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
    '''
    @communication
    Broadcast master's data to other peers in the same group
    group_unique has to be false
    '''
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
    '''
    @communication
    Reduct other peers' data to master in the group
    group_unique has to be false
    '''
    assert self.group_unique == False
    data = self.local_data
    if MVAPICH2:
      cache = garray.zeros(shape = self.local_data.shape, dtype = np.float32)
      self.group_comm.Reduce([tobuffer(data), MPI.FLOAT], [tobuffer(cache), MPI.FLOAT], root = 0)
      self.local_data = cache
    else:
      if self.group_rank == 0:
        cache = garray.empty(shape = (self.group_size, int(np.prod(self.local_data.shape))), dtype = np.float32)
        self.group_comm.Gather([tobuffer(data), MPI.FLOAT], [tobuffer(cache), MPI.FLOAT])
        for i in range(1, self.group_size):
          tmp  = garray.GPUArray(shape = self.local_data.shape, dtype = np.float32, gpudata = cache.ptr + cache.strides[0] * i)
          self.local_data += tmp
      else:
        self.group_comm.Gather([tobuffer(data), MPI.FLOAT], None)


  def _synchronize(self, communicator, data, num_worker):
    '''
    @communication.internal
    @param communicator:MPI world
    @param data:destination
    @param num_worker:number of workers in the world
    Add up all the local_data inside the communicator world to data
    '''
    if MVAPICH2:
      cache = garray.zeros(shape = data.shape, dtype = np.float32)
      communicator.Allreduce([tobuffer(data), MPI.FLOAT], [tobuffer(cache), MPI.FLOAT])
      garray.copy_to(cache, data)
    else:
      cache = garray.empty(shape = (num_worker, int(np.prod(data.shape))), dtype = np.float32)
      _ = time.time()
      communicator.Allgather([tobuffer(data), MPI.FLOAT], [tobuffer(cache), MPI.FLOAT])
      if INNER: MONITOR.add_comm(time.time() - _)
      _ = time.time()
      for i in range(1, num_worker):
        tmp  = garray.GPUArray(shape = data.shape, dtype = np.float32, gpudata = cache.ptr + cache.strides[0] * i)
        data += tmp
      if INNER: MONITOR.add_comm(time.time() - _)


  def group_synchronize(self):
    '''
    @communication
    Synchronize inside the group
    '''
    assert self.group_unique == False
    self._synchronize(self.group_comm, self.local_data, self.group_size)

  def master_synchronize(self):
    '''
    @communication
    Synchronize the masters
    '''
    if self.num_group == 1:
      return
    else:
      assert self.global_unique == False
      if self.group_rank == 0:
        self._synchronize(self.master_comm, self.local_data, self.num_group)


  def write(self, area, data, propagate = True, debug = False):
    '''
    @communication
    @param area:target area
    @param data:source data
    @param propagate:if True, sent data to other GPUs
    @param debug:if True, print out some debug information
    '''
    if self.num_group > 1 and self.group_size == 1 and propagate and not self.global_unique and debug:
      self.write_async(area, data)
      return

    if self.global_unique:
      if self.global_area.cmp(area):
        self.global_write(area, data, propagate)
      else:
        self.group_write(area, data, propagate)
    else:
      if not self.group_unique:
        self._partial_write(area, data)
        if propagate:
          if self.num_group == 1:
            self.group_synchronize()
          else:
            self.group_reduce()
            #self.master_write()
            #self.group_synchronize()
            self.master_synchronize()
            #if self.num_group != 1:
            self.group_bcast()
      else:
        self.group_write(area, data, propagate)

  def write_async(self, area, data):
    '''
    @communication.internal
    An asynchronousely wrint function designed specifically for weigt synchronization
    '''
    barrier(self.master_comm)
    self._async_flag = True

    send_data = [self.local_data] * self.num_group
    send_data[self.group_id] = None

    recv_data = [None] * self.num_group
    for i in range(self.num_group):
      if i == self.group_id: continue
      recv_data[i] = self.get_gpuarray(area)

    self._communicate(send_data, recv_data, self.global_comm, async = True)

  def check_param(self, other):
    '''
    @utility
    @param other:Compared VArray

    @return True if both global_slice_dim and group_slice_dim are the same
    '''
    return self.global_slice_dim == other.global_slice_dim and self.group_slice_dim == other.group_slice_dim

  def __add__(self, other):
    '''
    @computation

    This function is only called at FC layer, when adding bias up to output.
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

  def __mul__(self, other):
    '''
    @computation

    This function is only used in FC layer, when multiplying mask with output
    '''
    if np.isscalar(other):
      c = allocate_like(self)
      garray.copy_to(self.local_data * other, c.local_data)
      return c
    else:
      assert self.check_param(other)
      c = allocate_like(self)
      c.local_data  = self.local_data * other.local_data
      return c

  def __eq__(self, other):
    '''
    @computation

    This function is only called in logreg_cost when comparing maxid with label
    Since softmax layer always applys replica parallelism, so maxid should be group and global
    non-unique
    '''
    c = allocate_like(self)
    if isinstance(other, garray.GPUArray):
      if self.group_unique == False:
        assert other.shape == self.local_shape
        c.local_data = self.local_data == other
        return c
      else:
        c.local_data = self.local_data == other[self.local_area.slice]
      return c
    else:
      assert False

  def group_communicate(self):
    '''
    @communication.layer.internal

    Store group data into local_cache
    '''
    self.local_cache_area = Area.make_area(self.group_shape)
    if self.group_unique:
      self.local_cache = self.group_fetch(self.group_area)
    else:
      self.local_cache = self.local_data

  def global_communicate(self):
    '''
    @communication.layer

    Store global data into local_cache, used when the input of layer needs to be shared in model
    parallelism and replica parallelism
    '''
    self.local_cache_area = Area.make_area(self.global_shape)
    if self.global_unique:
      self.local_cache = self.fetch(self.local_cache_area)
    else:
      self.group_communicate()

  def channel_communicate(self, rank, slice_dim, padding = 0):
    '''
    @communication.layer
    @param rank:self rank
    @param slice_dim:The channel dimension
    @param padding:alwasy 0

    Store (num_channel / num_worker) of  data in local_cache, used when the input needs to be
    divided by channel at model parallelism
    '''
    if padding == 0:
      self.local_cache_area = Area.make_stripe_area(rank, slice_dim)
    else:
      tmp_area = Area.make_stripe_area(rank, slice_dim)
      if tmp_area._from[slice_dim] != 0:
        tmp_area._from[slice_dim] -= padding
      if tmp_area._to[slice_dim] != self.global_area._to[slice_dim]:
        tmp_area._to[slice_dim] += padding
      self.local_cache_area = tmp_area
    self.local_cache = self.fetch(self.local_cache_area)

  def batch_communicate(self, rank, slice_dim):
    '''
    @communication.layer
    @param rank:self rank
    @param slice_dim:batch dimension

    Store (num_batch / num_worker) of data in local_cache, used when the input needs to be divided
    by batch at data parallelism
    '''
    if slice_dim == self.group_slice_dim:
      self.local_cache_area = self.local_area
      self.local_cache = self.local_data
    else:
      self.local_cache_area = VArray.make_stripe_area(rank, slice_dim, self.group_area, self.group_size)
      self.local_cache = self.group_fetch(self.local_cache_area)

  def image_communicate(self, slice_dim, stride, filter_size, padding = 0, output_area = None):
    '''
    @communication.layer
    @param slice_dim:image height and width dimension, should be tuple
    @param stride:stride of sliding window
    @param filter_siz:window size
    @param padding:number of padding pixels need to add to final data
    @param output_area:the corresponding output area produced by the final data

    Store the necessary data in local, basically fetch overlapping data from other GPUs
    '''
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
      self.local_cache_area = Area(Point(*_from), Point(*_to))
      self.local_cache = self.group_fetch(self.local_cache_area, padding = padding, slice_dim = slice_dim)
    else:
      # norm layer, alwasy follow the distribution of previous layer
      # sounds like cheating
      self.cache_from_local_data()

  def cache_from_local_data(self):
    '''
    @utility.internal

    Set local_cache to local_data
    '''
    self.local_cache = copy.deepcopy(self.local_data)
    self.local_cache_area = self.local_area

  def get_pad_info(self, padding, old_shape, old_area, slice_dim = None):
    '''
    @utility.internal
    @param padding:padding
    @param old_shape:shape without padding
    @param old_area:area without padding
    @param slice_dim:image height and width, could be just height or just width

    @return new shape with padding and slice object of the new shape in old shape

    Used by merge to find the new shape and area for padding
    '''
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

  def unpad(self, data, padding, old_shape, old_area, slice_dim, debug = False):
    '''
    @utility
    @param data:data with padding that needs to be unpadded
    @param old_shape:shape with padding
    @param old_area:area with padding
    @param slice_dim:image height and width
    @param debug:if True, print some debug information

    @return GPUArray data without padding
    '''
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
    '''
    @utility
    @param scalar:fill array with this value
    '''
    self.local_data.fill(scalar)

  def get(self):
    '''
    @utility

    Get numpy ndarray of VArray
    '''
    if not self.global_unique and not self.group_unique:
      return self.local_data.get()
    else:
      return self.fetch(self.global_area).get()

  def printout(self, name, row_from = 0, row_to = 0, col_from = 0, col_to = 0):
    barrier(self.global_comm)
    if not self.group_unique and not self.global_unique:
      x = self.local_data
    else:
      x = self.fetch(self.global_area)

    if self.global_rank == 0:
      x.printout(name, row_from = row_from, row_to = row_to, col_from =  col_from, col_to = col_to)
    barrier(self.global_comm)

def array(obj, global_slice_dim = None, group_slice_dim = None, context = default_context):
  return VArray(array = obj,
                global_slice_dim = global_slice_dim,
                group_slice_dim = group_slice_dim,
                context = context)

def allocate(shape, global_slice_dim = None, group_slice_dim = None, context = default_context):
  return VArray(array = None,
                global_slice_dim = global_slice_dim,
                group_slice_dim = group_slice_dim,
                shape = shape,
                context = context)

def zeros(shape, global_slice_dim = None, group_slice_dim = None, context = default_context):
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

def random_uniform(shape, global_slice_dim = None, group_slice_dim = None, context = default_context):
  va = allocate(shape = shape,
                global_slice_dim = global_slice_dim,
                group_slice_dim = group_slice_dim,
                context = context)
  va.local_data = garray.random_uniform(va.local_shape)
  return va


def assemble(local_data, flat = False, axis = -1):
  assert len(local_data.shape) == 2 or len(local_data.shape) == 4
  assert axis == 0 or axis == -1
  if axis < 0: axis = len(local_data.shape) + axis

  shape_list = WORLD.allgather(local_data.shape)
  dim_sum = int(np.sum(np.array([x[axis] for x in shape_list])))

  if axis == 0:
    shape = tuple([dim_sum] + list(shape_list[0][1:]))
  else:
    shape = tuple(list(shape_list[0][:-1]) + [dim_sum])

  rst = allocate(shape = shape, global_slice_dim = None, group_slice_dim = axis)
  #print '%s, %s, %s' % (shape, rst.local_data.shape, local_data.shape)
  assert rst.local_data.shape == local_data.shape
  rst.local_data = local_data

  if flat:
    rst.gather()
    return rst.local_data
  else:
    return rst

def concatenate(labels):
  #concatenate labels in the order of rank
  cache = garray.empty(shape = (size, int(labels.size)), dtype = np.float32)
  WORLD.Allgather([tobuffer(labels), MPI.FLOAT], [tobuffer(cache), MPI.FLOAT])
  return cache.reshape((cache.size, ))
