from distbase.util import Assert
import garray
import varray
import copy
import numpy as np

import os

from multigpu import arr

def update(wts, grad, incr, epsilon, momentum, decay, batch_size):
  Assert.eq(grad.shape, wts.shape)

  assert(grad.dtype == np.float32)
  assert(wts.dtype == np.float32)
  assert(incr.dtype == np.float32)

  if momentum > 0.0:
    arr.matrix_add(incr, grad,
               alpha=momentum,
               beta=np.float32(epsilon / batch_size))

    arr.matrix_add(incr, wts,
               alpha=1,
               beta=np.float32(-decay * epsilon))

    arr.matrix_add(wts, incr)
  else:
    arr.matrix_add(wts, grad,
               alpha=1,
               beta=np.float32(epsilon / batch_size))

class Weight(object):
  def __init__(self, allocator):
    self._allocator = allocator

  def isweight(self):
    return self.name.startswith('weight')

  def to_gpu(self, obj):
    init = self._allocator.init_weight if self.isweight() else self._allocator.init_bias
    result = init(obj)
    return result

  def set_weight(self, w):
    if self.shape is None:
      self.shape = w.shape

    Assert.eq(w.shape, self.shape)
    self._wt = self.to_gpu(w)

  def set_grad(self, g):
    assert g.shape == self.shape
    assert g.dtype == np.float32
    self._grad = self.to_gpu(g)

  def set_incr(self, g):
    assert g.shape == self.shape
    assert g.dtype == np.float32
    self._incr = self.to_gpu(g)

  @property
  def grad(self):
    if self._grad is None or self._grad.shape != self.shape:
      init = self._allocator.init_weight if self.isweight() else self._allocator.init_bias
      self._grad = init(shape = shape)
    return self._grad

  @property
  def incr(self):
    if (self._incr is None or self._incr.shape != self.shape) and self.momentum > 0:
      init = self._allocator.init_weight if self.isweight() else self._allocator.init_bias
      self._incr = init(shape = shape)
      self._incr.fill(0.0)
    return self._incr

  @property
  def wt(self):
    if self._wt is not None:
      Assert.eq(self._wt.shape, self.shape)
    return self._wt

  def update(self, stat):
    return update(self.wt, self.grad, self.incr,
                  self.epsilon.get_value(stat), self.momentum, self.decay, stat.batch_size)

  def __repr__(self):
    return 'Weight(eps=%s mom=%s decay=%s)' % (self.epsilon, self.momentum, self.decay)

class WeightManager(object):
  def __init__(self):
    self._weights = []

  def __iter__(self):
    return iter(self._weights)

  def __getitem__(self, idx):
    return self._weights[idx]

  def clone(self):
    return copy.deepcopy(self)

  def empty(self, name, epsilon, momentum, decay, allocator):
    w = Weight(allocator = allocator)
    w.name = name
    w.shape = None
    w.decay = np.float32(decay)
    w.epsilon = epsilon
    w.momentum = np.float32(momentum)
    w._grad = w._wt = w._incr = None
    self._weights.append(w)
    return w

  def update(self, stat):
    for w in self._weights:
      update(w, stat)
WEIGHTS = WeightManager()
