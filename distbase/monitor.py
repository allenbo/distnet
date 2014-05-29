import numpy as np

class Monitor(object):
  COMP = 0
  COMM = 1
  def __init__(self):
    self._clear()
  
  def set_name(self, name):
    if name not in self._list:
      self._list[name] = [0.0, 0.0]
    self._curr_name = name

  def add_comp(self, elapsed):
    if self._curr_name is None:
      assert 'Have\'t specified the item name'
    
    name = self._curr_name
    self._list[name][Monitor.COMP] += elapsed
    self._comp += elapsed
  
  def add_comm(self, elapsed):
    if self._curr_name is None:
      assert 'Have\'t specified the item name'
    
    name = self._curr_name
    self._list[name][Monitor.COMM] += elapsed
    self._comm += elapsed

  def report(self):
    print '{:10}\t{:20}\t{:20}'.format('layer', 'comp', 'comm')
    for name in self._list:
      print '{:10}\t{:20}\t{:20}'.format(name, self._list[name][Monitor.COMP], self._list[name][Monitor.COMM])

    print '{:10}\t{:20}\t{:20}'.format('total', self._comp, self._comm)
    self._clear()

  def _clear(self):
    self._curr_name = None
    self._list = {}
    self._comp = 0
    self._comm = 0
MONITOR = Monitor()
