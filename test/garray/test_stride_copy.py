import garray
from pycuda import gpuarray
import numpy as np
import time
garray.device_init()

def test_small():
  a = garray.GPUArray((3, 3, 3, 3), dtype=np.float32)
  b = garray.GPUArray((3, 3, 3, 3), dtype=np.float32)
  a.fill(0)
  b.fill(1)
  a[:, 1:2, 1:2, :] = b[:, 1:2, 1:2, :]
  #print a


def test_stride_copy_4():
  n = np.zeros((96,55,55,128), dtype=np.float32)
  n[:] = 1

  z = np.zeros((96,55,55,128), dtype=np.float32)

  a = garray.GPUArray((96, 55, 55, 128), dtype=np.float32)
  b = garray.GPUArray((96, 55, 55, 128), dtype=np.float32)

  for slc in [
      np.index_exp[:, 0:24, 0:24, :],
      np.index_exp[:, 0:10, 0:10, :],
      np.index_exp[:, 0:5, 0:5, :],
      np.index_exp[:, 0:1, 0:1, :],
      np.index_exp[0:1, 0:1, 0:1, 0:1],
      ]:
    a.set(n)
    b.set(z)
   
    assert a[slc].get().sum() == n[slc].sum()
    assert b[slc].get().sum() == 0
    a[slc] = b[slc]
    assert a[slc].get().sum() == 0

    c = np.copy(n)
    c[slc] = 0
    
    assert a.get().sum() == c.sum(), (a.get().sum(), n.sum(), n[slc].sum())

  N = 10
  for slc in [
      np.index_exp[:, 0:24, 0:24, :],
      np.index_exp[:, 0:10, 0:10, :],
      np.index_exp[:, 0:5, 0:5, :],
      np.index_exp[:, 0:1, 0:1, :],
      np.index_exp[0:1, 0:1, 0:1, 0:1],
      ]:
    sz = np.prod(n[slc].shape) * 4
    st = time.time()
    for i in range(N):
      a[slc] = b[slc]
    ed = time.time()

    print '[%f] %d copies of %.2fMb in %f' % (1e-6 * N * sz / (ed - st), N, 1e-6 * sz, ed - st)
  
if __name__ == '__main__':
  test_small()
  test_stride_copy_4()
