import garray
import numpy as np
garray.device_init()

def test_1():
  a = np.random.randn(10).astype(np.float32)
  b = np.random.randn(5).astype(np.float32)
  c = np.random.randn(10).astype(np.float32)

  ga = garray.array(a)
  gb = garray.array(b)
  gc = garray.array(c)

  a = np.concatenate((a, b, c), axis = 0)
  ga = garray.concatenate((ga, gb, gc), axis = 0)

  assert np.abs(ga.get() - a).sum() == 0, a


def test_2():
  a = np.random.randn(10, 10).astype(np.float32)
  b = np.random.randn(10, 3).astype(np.float32)
  c = np.random.randn(10, 4).astype(np.float32)

  ga = garray.array(a)
  gb = garray.array(b)
  gc = garray.array(c)

  a = np.concatenate((a, b, c), axis = 1)
  ga = garray.concatenate((ga, gb, gc), axis = 1)

  assert np.abs(ga.get() - a).sum() == 0, a



def test_4():
  a = np.random.randn(96, 100, 100, 128).astype(np.float32)
  b = np.random.randn(96, 100, 10, 128).astype(np.float32)

  ga = garray.array(a)
  gb = garray.array(b)

  a = np.concatenate((a, b), axis = 2)
  ga = garray.concatenate((ga, gb), axis = 2)

  assert np.abs(ga.get() - a).sum() == 0, a

#test_4()
