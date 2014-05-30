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

def nonblock(send_data_list, recv_data_list):
  _ = time.time()
  send_req = []
  recv_req = []
  for i, data in enumerate(send_data_list):
    if data is None:continue
    send_req.append(WORLD.Isend([garray.tobuffer(data), MPI.FLOAT], dest = i))

  for i, data in enumerate(recv_data_list):
    if data is None:continue
    recv_req.append(WORLD.Irecv([garray.tobuffer(data), MPI.FLOAT], source = i))

  for req in send_req + recv_req: req.wait()

def test_point():
  buffer_sizes = [1, 8, 64, 512, 4096, 32 * 1024, 256 * 1024]

  for buffer_size in buffer_sizes:
    buffer_size *= 1024
    item_size = buffer_size / 4
    send_data_list = [None] * size
    recv_data_list = [None] * size

    for i in range(size):
      if i == rank:
        send_data_list[i] = garray.array(np.ones(item_size).astype(np.float32))
        recv_data_list[i] = garray.array(np.ndarray(item_size).astype(np.float32))

    _ = time.time()
    for i in range(iteration):
      nonblock(send_data_list, recv_data_list)
    elapsed = time.time() - _
    
    if rank == 0:
      real_time = elapsed / iteration
      theradical_time = buffer_size * (size - 1) * 2 / bandwidth
      print 'size:%d, real time:%f, theradical time:%f, ratio:%f' % (buffer_size, real_time, theradical_time, theradical_time / real_time * 100)

  MPI.Finalize()

if __name__ == '__main__':
  test_point()
