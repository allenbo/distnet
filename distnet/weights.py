from distnet.cuda_kernel import matrix_add
from distnet.util import Assert
import garray
import copy
import numpy as np


def update(wts, grad, incr, epsilon, momentum, decay, batch_size):
  #assert weight.incr.get().mean() < 1
  #a, b, c, = weight.incr.get().mean(), weight.wt.get().mean(), (grad.get() * weight.epsilon / batch_size).mean()
  #util.log_info('%s %s %s %s %s', weight.name, a, b, c, grad.get().mean())
  Assert.eq(grad.shape, wts.shape)
  
  assert(grad.dtype == np.float32)
  assert(wts.dtype == np.float32)
  assert(incr.dtype == np.float32)

  if momentum > 0.0:
    matrix_add(incr, grad, 
               alpha=momentum, 
               beta=np.float32(epsilon / batch_size))
    
    matrix_add(incr, wts, 
               alpha=1, 
               beta=np.float32(-decay * epsilon))
    
    matrix_add(wts, incr)
  else:
    matrix_add(wts, grad, 
               alpha=1, 
               beta=np.float32(epsilon / batch_size))
  
  #a2, b2, c2, = self.incr.get().mean(), self.wt.get().mean(), grad.get().mean()
  

def to_gpu(obj):
  if isinstance(obj, garray.GPUArray): 
    return obj
  
  assert obj.dtype == np.float32
  result = garray.array(obj, dtype = np.float32)
  assert result.dtype == np.float32
  return result


class Weight(object):
  def set_weight(self, w):
    if self.shape is None:
      self.shape = w.shape
    
    Assert.eq(w.shape, self.shape)
    self._wt = to_gpu(w)
    
  def set_grad(self, g):
    assert g.shape == self.shape
    assert g.dtype == np.float32
    self._grad = to_gpu(g)
    
  def set_incr(self, g):
    assert g.shape == self.shape
    assert g.dtype == np.float32
    self._incr = to_gpu(g)
    
  @property
  def grad(self):
    if self._grad is None or self._grad.shape != self.shape:
      self._grad = garray.zeros_like(self.wt)
      
    return self._grad
  
  @property
  def incr(self):
    if (self._incr is None or self._incr.shape != self.shape) and self.momentum > 0:
      self._incr = garray.zeros_like(self.wt)
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
    
  def empty(self, name, epsilon, momentum, decay):
    w = Weight()
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
