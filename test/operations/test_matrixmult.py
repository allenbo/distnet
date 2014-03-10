import garray
import random
import numpy as np
garray.device_init()

def test_matmult():
  m = random.randint(0, 1024)
  n = random.randint(0, 1024)
  k = random.randint(0, 1024)
  a = garray.array(np.random.randn(m, k).astype(np.float32))
  b = garray.array(np.random.randn(k, n).astype(np.float32))

  c = garray.matrixmult(a, b)

  na = a.get()
  nb = b.get()
  nc = np.dot(na, nb)

  diff = c.get() - nc
  diff = diff / nc
  assert (diff < 1e3).all()
  print 'Matrix passed the test'

test_matmult()
