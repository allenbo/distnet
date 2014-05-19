from distbase.util import Assert
import garray
import varray
import copy
import numpy as np

import os

from multigpu import uniformed_array, arr, allocate, zeros, default_context

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
  def __init__(self, group_slice_dim, context):
    self.group_slice_dim = group_slice_dim
    self.context = context
  
  def to_gpu(self, obj):
    assert obj.dtype == np.float32
    result = uniformed_array(obj, global_slice_dim = None, group_slice_dim = self.group_slice_dim, context = self.context)
    assert result.dtype == np.float32
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
      self._grad = allocate(shape = self.shape, global_slice_dim = None, group_slice_dim = self.group_slice_dim, context = self.context)
    return self._grad

  @property
  def incr(self):
    if (self._incr is None or self._incr.shape != self.shape) and self.momentum > 0:
      self._incr = zeros(shape = self.shape, global_slice_dim = None, group_slice_dim = self.group_slice_dim, context = self.context)
    return self._incr

  @property
  def wt(self):
    if self._wt is not None:
      Assert.eq(self._wt.shape, self.shape)
    return self._wt

  def update(self, batch_size):
    return update(self.wt, self.grad, self.incr,
                  self.epsilon, self.momentum, self.decay, batch_size)


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

  def empty(self, name, epsilon, momentum, decay, group_slice_dim = None, context = default_context):
    w = Weight(group_slice_dim = group_slice_dim, context = context)
    w.name = name
    w.shape = None
    w.decay = np.float32(decay)
    w.epsilon = np.float32(epsilon)
    w.momentum = np.float32(momentum)
    w._grad = w._wt = w._incr = None
    self._weights.append(w)
    return w

  def update(self, batch_size):
    for w in self._weights:
      update(w, batch_size)
WEIGHTS = WeightManager()
