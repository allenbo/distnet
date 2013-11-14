import varray as va
import numpy as np
from mpi4py import  MPI

WORLD = MPI.COMM_WORLD
rank = WORLD.Get_rank()

a = va.random.randn((4, 4) , slice_method = va.DistMethod.Square, slice_dim = (0, 1))
b = va.zeros_like(a)
print b.local_data.shape
area = va.Area(va.Point(0, 0), va.Point(3, 3))

c = np.ones((4, 4)).astype(np.float32)

b.write(area, c)
print b.local_data
#print b.local_data
'''
b.tmp_local_data = b.local_data
b.tmp_local_area = b.local_area
b.pad(1)
print b.tmp_local_data

b.tmp_local_data = b.unpad(b.tmp_local_data, 1)
print b.tmp_local_data
'''
