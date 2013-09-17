#!/usr/bin/env python

'''A relatively simple distributed network implementation, using async SGD.'''

from mpi4py import MPI
WORLD = MPI.COMM_WORLD

from fastnet import init_cuda
init_cuda.init(WORLD.Get_rank())

import ctypes
import numpy as np
import os

from fastnet import net, layer, data, parser
from fastnet.util import EZTimer

print 'CUDA', os.environ.get('MV2_USE_CUDA')

MASTER = 0
WORKERS = range(1, WORLD.Get_size())

batch_size = 128

data_dir = '/ssd/nn-data/cifar-10.old'
checkpoint_dir = './checkpoint'
param_file = 'config/cifar-10-18pct.cfg'

# train_range = range(101, 1301)
# test_range = range(1, 101)
#data_provider = 'imagenet'

data_provider = 'cifar10'
train_range = range(1, 41)
test_range = range(41, 49)

train_dp = data.get_by_name(data_provider)(data_dir,train_range)
test_dp = data.get_by_name(data_provider)(data_dir, test_range)

model = parser.parse_config_file(param_file)
net = net.FastNet(1.0, train_dp.image_shape + (batch_size, ), model)

class Tags(object):
  GRAD_SEND = 100
  WEIGHT_UPDATE = 200


def tobuffer(gpuarray):
  #print 'BUFFER: 0x%x' % gpuarray.ptr
  #print 'SIZE: %s, %s, %s' % (gpuarray.size, gpuarray.shape, gpuarray.dtype) 
  dtype = np.dtype(gpuarray.dtype)
  buf = ctypes.pythonapi.PyBuffer_FromReadWriteMemory(ctypes.c_long(gpuarray.ptr),
                                                      gpuarray.size * dtype.itemsize)
  return ctypes.cast(buf, ctypes.py_object).value

def wait_for_all(reqs):
  for r in reqs: r.Wait()

class Worker(object):
  def __init__(self):
    self.id = WORLD.Get_rank()
    
  def train(self):
    batch = train_dp.get_next_batch(batch_size)
    data, labels = net.prepare_for_train(batch.data, batch.labels)
    prediction = net.fprop(data)
    cost, correct = net.get_cost(labels, prediction)
    net.bprop(labels)
    self.send_grads()
    self.recv_weights()
    print cost, correct
    
  def send_grads(self):
    t = EZTimer('send grads')
    sends = []
    for idx, w in enumerate(layer.WEIGHTS):
      sends.append(WORLD.Isend(tobuffer(w.grad), dest=MASTER, tag=Tags.GRAD_SEND + idx))
    wait_for_all(sends)
    
  def recv_weights(self):
    t = EZTimer('recv weights')
    for idx, w in enumerate(layer.WEIGHTS):
      WORLD.Bcast(tobuffer(w.wt), root=MASTER)
      
    
  def run(self):
    while 1:
      self.train()
      self.send_grads()
      self.recv_weights()
      
    
class Master(object):
  def recv_grads(self, worker):
    recvs = []
    for idx, w in enumerate(layer.WEIGHTS):
      recvs.append(WORLD.Irecv(tobuffer(w.grad), source=worker, tag=Tags.GRAD_SEND + idx))
      
    wait_for_all(recvs)
    
  def update(self):
    layer.WEIGHTS.update(batch_size * len(WORKERS))
      
  def send_weights(self):
    # sends = []
    # for idx, w in enumerate(layer.WEIGHTS):
    #  sends.append(WORLD.ISend(tobuffer(w.weight), dest=worker, tag=Tags.WEIGHT_UPDATE + idx))
    # wait_for_all(sends)
    for idx, w in enumerate(layer.WEIGHTS):
      #print 'Sending...', idx
      WORLD.Bcast(tobuffer(w.wt), root=MASTER)
    
  def run(self):
    while 1:
      #print 'Fetching gradients...'
      for i in WORKERS:
        self.recv_grads(i)
        self.update()
      #print 'Sending weight updates...'
      self.send_weights()
      

if __name__ == '__main__':
  if WORLD.Get_rank() == 0:
    master = Master()
    master.run()
  else:
    worker = Worker()
    worker.run()