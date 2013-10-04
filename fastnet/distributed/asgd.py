#!/usr/bin/env python

'''A relatively simple distributed network implementation, using async SGD.'''

from fastnet import init_cuda, net, layer, data, parser, weights
from fastnet.util import EZTimer
from mpi4py import MPI
import ctypes
import numpy as np
import os
WORLD = MPI.COMM_WORLD

init_cuda.init(WORLD.Get_rank())

print 'CUDA', os.environ.get('MV2_USE_CUDA')

MASTER = 0
WORKERS = range(1, WORLD.Get_size())

batch_size = 128

data_dir = '/ssd/nn-data/imagenet/'
data_provider = 'imagenet'
checkpoint_dir = './checkpoint'
param_file = 'config/imagenet.cfg'

train_range = range(101, 1301)
test_range = range(1, 101)

data_provider = 'imagenet'
#train_range = range(1, 41)
#test_range = range(41, 49)

train_dp = data.get_by_name(data_provider)(data_dir,train_range)
test_dp = data.get_by_name(data_provider)(data_dir, test_range)

model = parser.parse_config_file(param_file)
network = net.FastNet((3, 224, 224, 1))
network = parser.load_model(network, model)

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
    data, labels = network.prepare_for_train(batch.data, batch.labels)
    prediction = network.fprop(data)
    cost, correct = network.get_cost(labels, prediction)
    network.bprop(labels)
    self.send_grads()
    self.recv_weights()
    print cost, correct
    
  def send_grads(self):
    _ = EZTimer('send grads')
    sends = []
    for idx, w in enumerate(layer.WEIGHTS):
      sends.append(WORLD.Isend(tobuffer(w.grad), dest=MASTER, tag=Tags.GRAD_SEND + idx))
    wait_for_all(sends)
    
  def recv_weights(self):
    _ = EZTimer('recv weights')
    
    for idx, w in enumerate(layer.WEIGHTS):
      WORLD.Recv(tobuffer(w.wt), source=MASTER, tag=Tags.WEIGHT_UPDATE + idx)
      
    
  def run(self):
    while 1:
      self.train()
      self.send_grads()
      self.recv_weights()
      
class WorkerProxy(object):
  def __init__(self, idx, wts):
    self.idx = idx
    self.wts = wts
    self.recvs = []
    
  def start_read(self):
    assert len(self.recvs) == 0 
    for idx, w in enumerate(self.wts):
      self.recvs.append(WORLD.Irecv(tobuffer(w.grad), source=self.idx, tag=Tags.GRAD_SEND + idx))
  
  def send_weights(self, wts):
    _ = EZTimer('send weights')
    for idx, w in enumerate(wts):
      WORLD.Send(tobuffer(w.wt), dest=self.idx, tag=Tags.WEIGHT_UPDATE + idx)
    
  def test(self):
    return np.all([r.Test() for r in self.recvs])
  
  def wait(self):
    [r.Wait() for r in self.recvs]
    self.recvs = []
  
  def try_fetch(self):
    if len(self.recvs) == 0:
      self.start_read()
    
    if not self.test():
      return False
    
    self.wait()
    self.start_read()
    return True
  
      
class Master(object):
  def __init__(self):
    self._workers = {}
    self._master_wts = layer.WEIGHTS
    self._requests = []
    
    for w in WORKERS:
      self._workers[w] = WorkerProxy(w, layer.WEIGHTS.clone())
      
  def update(self, worker_wts):
    _ = EZTimer('update')
    for idx, worker_wt in enumerate(worker_wts):
      master_wt = self._master_wts[idx]
      weights.update(master_wt.wt, 
                     worker_wt.grad,
                     master_wt.incr,
                     master_wt.epsilon,
                     master_wt.momentum,
                     master_wt.decay,
                     128)
    
  def run(self):
    while 1:
      #print 'Fetching gradients...'
      for w in self._workers.values():
        if w.try_fetch():
          self.update(w.wts)
          w.send_weights(self._master_wts)
      
      #print 'Sending weight updates...'
      

if __name__ == '__main__':
  if WORLD.Get_rank() == 0:
    master = Master()
    master.run()
  else:
    worker = Worker()
    worker.run()
