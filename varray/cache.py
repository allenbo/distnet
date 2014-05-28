import garray
import numpy as np
import random

class Cache(object):
  DEFAULT_CAPACITY = 200 * 1024 * 1024
  def __init__(self, capacity = DEFAULT_CAPACITY):
    self._cache_pool = {}
    self.max_capacity = capacity
    self.curr_capacity = 0

  def get(self, area):
    shape = area.shape
    if shape in self._cache_pool:
      return self._cache_pool[shape]
    else:
      size = int( np.prod(shape) )
      array = garray.GPUArray(shape, dtype = np.float32)
      if size > self.max_capacity: 
        return array
      
      while size + self.curr_capacity > self.max_capacity:
        shape = random.choice(self._cache_pool.keys())
        self.curr_capacity -= np.prod(shape)
        del self._cache_pool[shape]

      self._cache_pool[shape] = array
      self.curr_capacity += size
      return array

  def clear(self):
    self.curr_capacity = 0
    self._cache_pool = {}
