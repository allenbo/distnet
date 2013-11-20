import garray
import numpy as np


def test_1():
  stop = 53
  a = np.random.randn(100).astype(np.float32)
  ga = garray.array(a)

  b = np.random.randn(10).astype(np.float32)
  gb = garray.array(b)
  a[1:20:2] = b
  ga[1:20:2] = gb
  print ga.get() - a

def test_2():
  a = np.random.randn(10000, 10000).astype(np.float32)
  ga = garray.array(a)
  b = np.random.randn(100, 100).astype(np.float32)
  gb = garray.array(b)
  a[2:102, 3:103] = b
  ga[2:102, 3:103] = gb

  print ga.get() - a


def test_3():
  a = np.random.randn(3, 224,224).astype(np.float32)
  ga = garray.array(a)
  b = np.random.randn(3, 51, 51).astype(np.float32)
  gb = garray.array(b)
  a[:, 10:112:2, 10:112:2] = b
  ga[:, 10:112:2, 10:112:2] = gb

  print ga.get() - a


def test_4():
  a = np.random.randn(96, 32, 32, 128).astype(np.float32)
  ga = garray.array(a)
  b = np.random.randn(96, 16, 16, 128).astype(np.float32)
  gb = garray.array(b)

  a[:, 0:16, 0:16, :] = b

  ga[:, 0:16, 0:16, :] = gb

  print ga.get() - a

test_4()
