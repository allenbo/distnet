import numpy as np
import sys

class Monitor(object):
  COMP = 0
  COMM = 1
  MARSHALL = 2
  MERGE = 3
  def __init__(self, name, active = False):
    self._name = name
    self._active = active
    self._clear()
  
  def set_name(self, name):
    if self._active == False: return
    if name not in self._list:
      self._list[name] = [0.0, 0.0, 0.0, 0.0]
    self._curr_name = name

  def add_comp(self, elapsed):
    if self._active == False: return
    if self._curr_name is None:
      assert 'Have\'t specified the item name'
    
    name = self._curr_name
    self._list[name][Monitor.COMP] += elapsed
    self._comp += elapsed
  
  def add_comm(self, elapsed):
    if self._active == False: return
    if self._curr_name is None:
      assert 'Have\'t specified the item name'
    
    name = self._curr_name
    self._list[name][Monitor.COMM] += elapsed
    self._comm += elapsed

  def add_marshall(self, elapsed):
    if self._active == False: return
    if self._curr_name is None:
      assert 'Have\'t specified the item name'
    
    name = self._curr_name
    self._list[name][Monitor.MARSHALL] += elapsed
    self._marshall += elapsed

  def add_merge(self, elapsed):
    if self._active == False: return
    if self._curr_name is None:
      assert 'Have\'t specified the item name'
    
    name = self._curr_name
    self._list[name][Monitor.MERGE] += elapsed
    self._merge += elapsed

  def report(self):
    if self._active == False: return
    format = '{:10}\t{:20}\t{:20}\t{:20}\t{:20}\t{:20}'
    print >> sys.stderr, '-' * 8 + self._name + '-' * 8
    print >> sys.stderr, format.format('layer', 'comp', 'comm', 'marshal', 'merge', 'sum')
    for name in self._list:
      print >> sys.stderr, format.format(name, self._list[name][Monitor.COMP],
          self._list[name][Monitor.COMM], self._list[name][Monitor.MARSHALL],
          self._list[name][Monitor.MERGE], np.sum(self._list[name]))

    print >> sys.stderr, format.format('total', self._comp, self._comm, self._marshall,
        self._merge, self._comp + self._comm + self._marshall, self._merge)
    self._clear()

  def _clear(self):
    self._curr_name = None
    self._list = {}
    self._comp = 0
    self._comm = 0
    self._marshall = 0
    self._merge = 0

MONITOR = Monitor('default')
