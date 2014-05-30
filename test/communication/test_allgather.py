import garray
import numpy as np
from mpi4py import MPI
import time
garray.device_init()

WORLD = MPI.COMM_WORLD
size = WORLD.Get_size()
rank = WORLD.Get_rank()

iteration = 3
bandwidth = 5.0 * 1e9 

def test_point():
  buffer_sizes = [1, 8, 64, 512, 4096, 32 * 1024, 256 * 1024]

  for buffer_size in buffer_sizes:
    buffer_size *= 1024
    item_size = buffer_size / 4
    data = garray.array(np.random.randn(item_size).astype(np.float32))
    cache = garray.empty(shape = (size, int(np.prod(data.shape))), dtype = np.float32)

    _ = time.time()
    for i in range(iteration):
      WORLD.Allgather([garray.tobuffer(data), MPI.FLOAT], [garray.tobuffer(cache), MPI.FLOAT])
    elapsed = time.time() - _
    
    if rank == 0:
      real_time = elapsed / iteration
      theradical_time = buffer_size * (size - 1) * 2 / bandwidth
      print 'size:%d, real time:%f, theradical time:%f, ratio:%f' % (buffer_size, real_time, theradical_time, theradical_time / real_time * 100)

  MPI.Finalize()

if __name__ == '__main__':
  test_point()
