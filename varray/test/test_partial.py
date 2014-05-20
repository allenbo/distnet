from varray.ndarray import VArray, DistMethod
from varray.context import Context
import numpy as np
import garray
garray.device_init()

shape = (128, 1)
hosta = np.ones(shape = shape).astype(np.float32)
virtuala = VArray(hosta,
                  global_slice_dim = None,
                  group_slice_dim = None
                  )
print virtuala.local_area

#shape = (3, 224, 224, 128)
#hosta = np.ones(shape = shape).astype(np.float32)
#global_slice_dim = 3
#group_slice_dim = (1, 2)
#
#worker_group = [4, 4]
#context = Context(worker_group)
#
#virtuala = VArray(hosta,
#                  global_unique = True,
#                  group_unique = True,
#                  context = context,
#                  global_slice_dim = global_slice_dim,
#                  group_slice_dim = group_slice_dim
#                  )
#
#print 'global_shape', virtuala.global_shape
#print 'global_area', virtuala.global_area
#
#print 'group_shape', virtuala.group_shape
#print 'group_area', virtuala.group_area
#
#print 'local_shape', virtuala.local_shape
#print 'local_area', virtuala.local_area
#
#if virtuala.group_rank != 0:
#  virtuala.fill(0)
#
#virtuala.group_bcast()
#assert (virtuala.local_data.get() == hosta).all()
#print 'group_bcast passed'
#
#virtuala.fill(0)
#gpua = garray.array(hosta)
#virtuala.group_write(area = virtuala.local_area, data = gpua)
#assert (virtuala.local_data.get() == hosta * virtuala.group_size).all()
#print 'group_write passed'
#
#virtuala.master_write()
#if virtuala.group_rank == 0:
#  assert (virtuala.local_data.get() == hosta * virtuala.global_size).all()
#
#virtuala.global_comm.Barrier()
#
#virtuala.fill(1.0)
#virtuala.group_reduce()
#if virtuala.group_rank == 0:
#  assert (virtuala.local_data.get() == hosta * virtuala.group_size).all()
#print 'group_reduce passed'
#virtuala.fill(0)
#virtuala.write(area = virtuala.local_area, data = gpua)
#
#assert (virtuala.local_data.get() == hosta * virtuala.global_size).all()
#print 'global write passed'
