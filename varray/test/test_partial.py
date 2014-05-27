from varray.ndarray import VArray, DistMethod
from varray.context import Context
import numpy as np
import garray
from mpi4py import MPI
garray.device_init()

#shape = (128, 1)
#hosta = np.ones(shape = shape).astype(np.float32)
#virtuala = VArray(hosta,
#                  global_slice_dim = None,
#                  group_slice_dim = None
#                  )
#print virtuala.local_area

#shape = (3, 16, 16, 32)
shape = (3, 4)
hosta = np.arange(np.prod(shape)).astype(np.float32).reshape(shape)
global_slice_dim = None
group_slice_dim = None

worker_group = [2, 2]
context = Context(worker_group)

virtuala = VArray(hosta,
                  context = context,
                  global_slice_dim = global_slice_dim,
                  group_slice_dim = group_slice_dim
                  )

#assert (virtuala.local_data.get() == hosta[virtuala.local_area.slice]).all()
#virtuala.printout('virtual')
#assert (x.get() == hosta[virtuala.group_area.slice]).all()


print 'global_shape', virtuala.global_shape
print 'global_area', virtuala.global_area

print 'group_shape', virtuala.group_shape
print 'group_area', virtuala.group_area

print 'local_shape', virtuala.local_shape
print 'local_area', virtuala.local_area

if virtuala.group_rank != 0:
  virtuala.fill(0)

virtuala.group_bcast()
assert (virtuala.local_data.get() == hosta).all()
print 'group_bcast passed'

virtuala.fill(0)
gpua = garray.array(hosta)
virtuala.group_write(area = virtuala.local_area, data = gpua)
assert (virtuala.local_data.get() == hosta * virtuala.group_size).all()
print 'group_write passed'

virtuala.master_write()
if virtuala.group_rank == 0:
  assert (virtuala.local_data.get() == hosta * virtuala.global_size).all()
virtuala.global_comm.Barrier()
print 'master_write passed'

virtuala.fill(1.0)
host = virtuala.local_data.get()
virtuala.group_reduce()
if virtuala.group_rank == 0:
  assert (virtuala.local_data.get() == host * virtuala.group_size).all()
print 'group_reduce passed'

virtuala.fill(1.0)
host = virtuala.local_data.get()
virtuala.group_synchroize()
assert (virtuala.local_data.get() == host * virtuala.group_size).all()
print 'group_synchronize'

MPI.Finalize()
#virtuala.fill(0)
#virtuala.write(area = virtuala.local_area, data = gpua)
#
#assert (virtuala.local_data.get() == hosta * virtuala.global_size).all()
#print 'global write passed'
