import numpy as np
import sys

class Monitor(object):
  FPROP = 0
  BPROP = 1
  WPROP = 2
  UPDATE = 3
  COMM = 4
  MARSHAL = 5
  MERGE = 6
  LEN = 7

  def __init__(self, name, active = False):
    self._name = name
    self._active = active
    self._clear()

  def set_name(self, name):
    if self._active == False: return
    if name not in self._list:
      self._list[name] = [0.0] * Monitor.LEN
    self._curr_name = name

  def get_name(self):
    if self._active == False: return None
    return self._curr_name

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
    self._list[name][Monitor.MARSHAL] += elapsed
    self._marshall += elapsed

  def add_merge(self, elapsed):
    if self._active == False: return
    if self._curr_name is None:
      assert 'Haven\'t specified the item name'

    name = self._curr_name
    self._list[name][Monitor.MERGE] += elapsed
    self._merge += elapsed

  def add_fprop(self, elapsed):
    if self._active == False: return
    if self._curr_name is None:
      assert 'Haven\'t specified the item name'

    name = self._curr_name
    self._list[name][Monitor.FPROP] += elapsed
    self._fprop += elapsed

  def add_bprop(self, elapsed):
    if self._active == False: return
    if self._curr_name is None:
      assert 'Haven\'t specified the item name'

    name = self._curr_name
    self._list[name][Monitor.BPROP] += elapsed
    self._bprop += elapsed

  def add_wprop(self, elapsed):
    if self._active == False: return
    if self._curr_name is None:
      assert 'Haven\'t specified the item name'

    name = self._curr_name
    self._list[name][Monitor.WPROP] += elapsed
    self._wprop += elapsed

  def add_update(self, elapsed):
    if self._active == False: return
    if self._curr_name is None:
      assert 'Haven\'t specified the item name'

    name = self._curr_name
    self._list[name][Monitor.UPDATE] += elapsed
    self._update += elapsed

  def report(self):
    if self._active == False: return
    format = '{:10}' + '\t{:20}' * (Monitor.LEN  + 1)
    print >> sys.stderr, '-' * 8 + self._name + '-' * 8
    print >> sys.stderr, format.format('layer', 'fprop', 'bprop', 'wprop', 'update', 'comm', 'marshal', 'merge', 'sum')
    for name in self._list:
      l = self._list[name]
      print >> sys.stderr, format.format(name, l[Monitor.FPROP], l[Monitor.BPROP], l[Monitor.WPROP],
              l[Monitor.UPDATE], l[Monitor.COMM], l[Monitor.MARSHAL], l[Monitor.MERGE], np.sum(self._list[name]))

    print >> sys.stderr, format.format('total', self._fprop, self._bprop, self._wprop, self._update, self._comm, self._marshall,
        self._merge, self._fprop + self._bprop + self._wprop + self._update + self._comm + self._marshall, self._merge)
    self._clear()

  def _clear(self):
    self._curr_name = None
    self._list = {}
    self._comm = 0
    self._marshall = 0
    self._merge = 0
    self._fprop = 0
    self._bprop = 0
    self._wprop = 0
    self._update = 0

MONITOR = Monitor('default')
