import garray
from garray import backend, reshape_first, reshape_last
import numpy as np
import time
import random
garray.device_init()


def get_slices(shape):
  slices = []
  new_shape = []
  for k in range(len(shape)):
    fd = random.randint(0, shape[k])
    sd = random.randint(0, shape[k])
    while sd == fd:
      sd = random.randint(0, shape[k])
    new_shape.append(abs(fd - sd))
    slices.append(slice(min(fd, sd), max(fd, sd)))
  return slices, tuple(new_shape)

# array test
def test_array():
  shape = []
  for i in range(4):
    shape.append(random.randint(1, 100))
    a = np.ndarray(shape = shape, dtype = np.float32)
    
    ga = garray.array(a)
    assert ga.shape == a.shape
    ga = garray.array(a, to2dim = True)
    if len(shape) > 2:
      assert len(ga.shape) == 2


# new getitem
def test_new_getitem():
  shape = []
  for i in range(4):
    shape.append(random.randint(10, 100))
    ha = np.random.randn(*shape).astype(np.float32)
    ga = garray.array(ha)

    slices, _ = get_slices(shape) 

    gb = ga[tuple(slices)]
    hb = ha[tuple(slices)]
    assert (hb == gb.get()).all()

# new_setitem
def test_new_setitem():
  shape = []
  for i in range(4):
    shape.append(random.randint(10, 100))
    ha = np.random.randn(*shape).astype(np.float32)
    ga = garray.array(ha)

    slices, new_shape = get_slices(shape)
    
    hb = np.random.randn(*new_shape).astype(ha.dtype)
    gb = garray.array(hb)

    ga[tuple(slices)] = gb
    ha[tuple(slices)] = hb
    assert (ha == ga.get()).all()

def test_setitem_sum():
  shape = []
  for i in range(4):
    shape.append(random.randint(10, 100))

  slices, new_shape = get_slices(shape)
  ha = np.random.randn(*shape).astype(np.float32)
  hb = np.random.randn(*new_shape).astype(np.float32)
  ga = garray.array(ha)
  gb = garray.array(hb)

  ga.setitem_sum(tuple(slices), gb)
  ha[slices] += hb
  
  assert (ha == ga.get()).all()
  

# newadd
def test_newadd():
  shape = []
  for i in range(4):
    shape.append(random.randint(10, 100))

  ha = np.random.randn(*shape).astype(np.float32)
  ga = garray.array(ha)

  # when right hand operator is 2D array
  hb = np.random.randn(shape[0], 1).astype(np.float32)
  gb = garray.array(hb)
  
  gc = ga + gb
  hc = (reshape_first(ha) + hb).reshape(tuple(shape))
  diff = hc - gc.get()
  assert (diff < 1e-5).all()

  hb = np.random.randn(1, shape[-1]).astype(np.float32)
  gb = garray.array(hb)

  gc = ga + gb
  hc = (reshape_last(ha) + hb).reshape(tuple(shape))
  diff = hc - gc.get()
  assert (diff < 1e-4).all()

  # when right hand operator is 4D array
  hb = np.random.randn(*shape).astype(np.float32)
  gb = garray.array(hb)

  gc = ga + gb
  hc = ha + hb
  diff = hc - gc.get()
  assert (diff < 1e-4).all()

# newsub
def test_newsub():
  shape = []
  for i in range(4):
    shape.append(random.randint(10, 100))

  ha = np.random.randn(*shape).astype(np.float32)
  ga = garray.array(ha)

  # when right hand operator is 2D array
  hb = np.random.randn(shape[0], 1).astype(np.float32)
  gb = garray.array(hb)
  
  gc = ga - gb
  hc = (reshape_first(ha) - hb).reshape(tuple(shape))
  diff = hc - gc.get()
  assert (diff < 1e-5).all()

  hb = np.random.randn(1, shape[-1]).astype(np.float32)
  gb = garray.array(hb)

  gc = ga - gb
  hc = (reshape_last(ha) - hb).reshape(tuple(shape))
  diff = hc - gc.get()
  assert (diff < 1e-4).all()

  # when right hand operator is 4D array
  hb = np.random.randn(*shape).astype(np.float32)
  gb = garray.array(hb)

  gc = ga - gb
  hc = ha - hb
  diff = hc - gc.get()
  assert (diff < 1e-4).all()
# newdiv
def test_newdiv():
  shape = []
  for i in range(4):
    shape.append(random.randint(10, 100))

  ha = np.random.randint(20, size = shape).astype(np.float32) + 1.0
  ga = garray.array(ha)

  # when right hand operator is 2D array
  hb = np.random.randint(20, size = (shape[0], 1)).astype(np.float32) + 1.0
  gb = garray.array(hb)
  
  gc = ga / gb
  hc = (reshape_first(ha) / hb).reshape(tuple(shape))
  diff = hc - gc.get()
  assert (diff < 1e-5).all()

  hb = np.random.randint(20, size = (1, shape[-1])).astype(np.float32) + 1.0
  gb = garray.array(hb)

  gc = ga / gb
  hc = (reshape_last(ha) / hb).reshape(tuple(shape))
  diff = hc - gc.get()
  assert (diff < 1e-4).all()

  # when right hand operator is 4D array
  hb = np.random.randint(20, size = shape).astype(np.float32) + 1.0
  gb = garray.array(hb)

  gc = ga / gb
  hc = ha / hb
  diff = hc - gc.get()
  assert (diff < 1e-4).all()

def test_sum():
  shape = []
  for i in range(2):
    shape.append(random.randint(10, 1000))
  
  ha = np.random.randint(20, size = shape).astype(np.float32)
  ga = garray.array(ha)

  gb = garray.sum(ga, axis = 0)
  hb = ha.sum(axis = 0)

  diff = abs(gb.get() - hb)
  assert (diff == 0).all()

  gb = garray.sum(ga, axis = 1)
  hb = ha.sum(axis = 1)
  
  diff = abs(gb.get().flatten() - hb.flatten())
  assert (diff < 1e-4).all()

def test_partial_copy():
  shape = []
  for i in range(2):
    shape.append(random.randint(100, 1000))

  ha = np.random.randn(*shape).astype(np.float32)
  ga = garray.array(ha)

  f = random.randint(0, shape[1])
  while True:
    t = random.randint(0, shape[1])
    if t != f:
      break
  gb = garray.partial_copy(ga, min(f, t), max(f, t))
  hb = ha[:, min(f, t):max(f, t)]

  assert (hb == gb.get()).all()

def test_max():
  shape = []
  for i in range(2):
    shape.append(random.randint(10, 1000))
  ha = np.random.randn(*shape).astype(np.float32)
  ga = garray.array(ha)

  gb = garray.max(ga, axis = 0)
  hb = np.max(ha, axis = 0)

  assert (hb.flatten() == gb.get().flatten()).all()

  gb = garray.max(ga, axis = 1)
  hb = np.max(ha, axis = 1)

  assert (hb.flatten() == gb.get().flatten()).all()

def test_argmax():
  shape = []
  for i in range(2):
    shape.append(random.randint(10, 1000))
  ha = np.random.randn(*shape).astype(np.float32)
  ga = garray.array(ha)

  gb = garray.argmax(ga, axis = 0)
  hb = np.argmax(ha, axis = 0)

  assert (hb.flatten() == gb.get().flatten()).all()

  gb = garray.argmax(ga, axis = 1)
  hb = np.argmax(ha, axis = 1)

  assert (hb.flatten() == gb.get().flatten()).all()

if __name__ == '__main__':
  test_array()
  test_new_getitem()
  test_new_setitem()
  test_setitem_sum()
  test_newadd()
  test_newsub()
  test_newdiv()
  test_sum()
  test_partial_copy()
  test_max()
  test_argmax()
