import cudaconv
from mpi4py import MPI
cudaconv.init(MPI.COMM_WORLD.Get_rank())

import varray
import numpy as np
import cProfile
p = cProfile.Profile()
p.enable()

from varray import rank, MASTER, util

def mlog(*args, **kw):
  if rank == MASTER:
    util.log(*args, **kw)

#a = va.random.square_randn((3, 100, 100, 3) , slice_dim = (1, 2))
#mlog('%s', a.local_shape)

#area = va.Area(va.Point(0, 45, 45, 0), va.Point(2,99, 99, 2))
'''
for i in range(1000):
  rst = a.fetch(area)
  mlog('%s', rst.shape)
  #print rst.shape

p.disable()
p.dump_stats('./profile.%d' % rank)
'''
#a = va.random.randn((4,4))
#print a.local_shape
#print a.local_data
#
#a.gather()
#print a.local_shape
#print a.local_data
print '*'* 10 , rank, '*' * 10
np.random.seed(0)
a = np.random.randn(4, 4).astype(np.float32)
print a

tmp =  varray.array(a, slice_dim = (0, 1))
data = tmp.fetch(varray.Area(varray.Point(0,0), varray.Point(2, 2)), padding = -1)
print data
#print va.local_data.get() - a
