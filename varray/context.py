from mpi4py import MPI
import numpy as np

WORLD = MPI.COMM_WORLD
MASTER = 0
size = WORLD.Get_size()
rank = WORLD.Get_rank()

class Context(object):
  def __init__(self, workers_group):
    self._global_comm = MPI.COMM_WORLD
    self._global_group = self._global_comm.Get_group()
    self._global_rank = self._global_comm.Get_rank()
    self._global_size = self._global_comm.Get_size()

    self._workers_group = workers_group[:]
    self._num_group = len(self._workers_group)
    if self._num_group == 0:
      self._num_group = 1
      self._workers_group = [self._global_size]
    assert np.sum(self._workers_group) <= self._global_size, '%s' % (self._workers_group)
    self._comms = []
    self._group_master = []

    self._create_new_comm()

  def _create_new_comm(self):
    self._group_master.append(0)
    worker_rank_accum = self._workers_group[:]
    for i in range(len(self._workers_group)):
      if i == 0:
        continue
      self._group_master.append(worker_rank_accum[i-1])
      worker_rank_accum[i] += worker_rank_accum[i-1]

    for i in range(len(self._workers_group)):
      start = 0 if i == 0 else worker_rank_accum[i-1]
      group_ranks = range(start, worker_rank_accum[i])
      group = self._global_group.Incl(group_ranks)
      
      new_comm = self._global_comm.Create(group)
      self._comms.append(new_comm)
      if self.global_rank in group_ranks:
        self._group_comm = new_comm
        self._group_id = i

    self._group_rank = self._group_comm.Get_rank()
    self._group_size = self._group_comm.Get_size()

    if self._num_group != 1:
      group = self._global_group.Incl(self._group_master)
      master_comm = self._global_comm.Create(group)
      if self._group_rank == 0:
        self._master_comm = master_comm
      else:
        self._master_comm = None
    else:
      self._master_comm = None


    print 'GlobalRank:%d, GroupId:%d, GroupRank:%d, GroupSize:%d, GroupMasters:%s' % (self._global_rank,
        self._group_id, self._group_rank, self._group_size, self._group_master)
  
  @property
  def num_group(self): return self._num_group

  @property
  def group_id(self): return self._group_id

  @property
  def global_comm(self): return self._global_comm
  
  @property
  def group_comm(self): return self._group_comm
  
  @property
  def master_comm(self): return self._master_comm
  @property
  def global_rank(self): return self._global_rank

  @property
  def group_rank(self): return self._group_rank

  @property
  def global_size(self): return self._global_size

  @property
  def group_size(self): return self._group_size

  @property
  def group_master(self): return self._group_master


  def get_group_id(self, global_rank):
    if self._num_group == 1:
      return 0
    
    start, stop = 0, 0
    for i in range(len(self._workers_group)):
      stop += self._workers_group[i]
      if start <= global_rank < stop:
        return i
      start = stop

  def get_group_rank(self, global_rank):
    if self._num_group == 1: return global_rank
    group_id = self.get_group_id(global_rank)
    return global_rank - self._group_master[group_id]


default_context = Context([])
