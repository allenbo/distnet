import json
from pycuda import gpuarray, driver, autoinit
import numpy as np
import time
import cudaconv
import garray
import cPickle as pickle

executer_dict = {}
def register_executer(name, _class):
  if name in executer_dict:
    print 'Executer', name, 'already registered'
  else:
    executer_dict[name] = _class

def get_executer(name):
  if name not in executer_dict:
    import sys
    print >> sys.stderr, 'NO %s executer' % name
    sys.exit(-1)
  else:
    return executer_dict[name]

class Executer(object):
  def __init__(self, decr, param):
    self.count = 3
    self.decr = decr
    self.param = param
    self.num_test = self.decr['tests']
    self.num_worker = self.decr['workers']
    self.state = self.decr['state']
    self.type = self.decr['type']

  def execute(self):
    assert False, 'Implementation needed'

class ConvExecuter(Executer):
  def execute(self):
    times = []
    for param in self.param:
      input_shape = tuple(param['input_shape'])
      output_shape = tuple(param['output_shape'])
      filter_shape = tuple(param['filter_shape'])
      if filter_shape[-1] % 16 != 0:
        num_filter = (filter_shape[-1] + 16 - 1) / 16 * 16
        filter_shape = filter_shape[:-1] + (num_filter, )
        print 'Change the number of filters to %d and  make it a multiple of 16' % num_filter
        output_shape = (num_filter, ) + output_shape[1:] 

      input = gpuarray.to_gpu(np.random.randn(*input_shape).astype(np.float32))
      output = gpuarray.to_gpu(np.random.randn(*output_shape).astype(np.float32))
      filter = gpuarray.to_gpu(np.random.randn(*filter_shape).astype(np.float32))
       
      ingrad = gpuarray.to_gpu(np.random.randn(*output_shape).astype(np.float32))
      outgrad = gpuarray.to_gpu(np.random.randn(*input_shape).astype(np.float32))

      image_y = input_shape[1]
      image_x = input_shape[2]
      output_y = output_shape[1]
      output_x = output_shape[2]
      padding = param['padding']
      stride = param['stride']
      channel = filter_shape[0]
      filter_size = filter_shape[1]

      cudaconv.convFilterActs(input, filter, output, image_y, output_y, output_x, -padding, stride,
          channel, 1)
      cudaconv.convImgActs(ingrad, filter, outgrad, image_y, image_x, output_y,
          -padding, stride, channel, 1)
      cudaconv.convWeightActs(input, ingrad, filter, image_y, output_y, output_x, filter_size,
          -padding, stride, channel, 1, 0)

      driver.Context.synchronize()

      start = time.time()
      for i in range(self.count):
        cudaconv.convFilterActs(input, filter, output, image_y, output_y, output_x, -padding, stride,
            channel, 1)
        cudaconv.convImgActs(ingrad, filter, outgrad, image_y, image_x, output_y,
            -padding, stride, channel, 1)
        cudaconv.convWeightActs(input, ingrad, filter, image_y, output_y, output_x, filter_size,
            -padding, stride, channel, 1, 0)
        driver.Context.synchronize()
      
      elapsed = (time.time() - start) / self.count
      times.append(elapsed)
    
    if self.num_test == 0:
      return 100.0
    if self.num_test != 1:
      assert self.type == 'max'
      return max(times)
    return times[0]


class PoolExecuter(Executer):
  def execute(self):
    times = []
    for param in self.param:
      input_shape = tuple(param['input_shape'])
      output_shape = tuple(param['output_shape'])

      if input_shape[0] % 16 != 0:
        num_filter = (input_shape[0] + 16 -1 ) / 16 * 16
        input_shape = (num_filter, ) + input_shape[1:]
        output_shape = (num_filter, ) + output_shape[1:]
        print 'Change the number of filters to %d and  make it a multiple of 16' % num_filter
      
      input = gpuarray.to_gpu(np.random.randn(*input_shape).astype(np.float32))
      outgrad = gpuarray.to_gpu(np.random.randn(*input_shape).astype(np.float32))
      output = gpuarray.to_gpu(np.random.randn(*output_shape).astype(np.float32))
      ingrad = gpuarray.to_gpu(np.random.randn(*output_shape).astype(np.float32))

      input_y = input_shape[1]
      input_x = input_shape[2]
      output_y = output_shape[1]
      output_x = output_shape[2]

      channel = input_shape[0]
      pool_size = param['pool_size']
      start = param['start']
      stride = param['stride']
      
      cudaconv.convLocalMaxPool(input, output, channel, pool_size, start, stride, input_y,
          output_y, output_x)
      cudaconv.convLocalMaxUndo(input, ingrad, output, outgrad, pool_size, start, stride, output_y,
          output_x, input_y)

      driver.Context.synchronize()
      s = time.time()
      for i in range(self.count):
        # forward
        cudaconv.convLocalMaxPool(input, output, channel, pool_size, start, stride, input_y,
            output_y, output_x)
        # backward
        cudaconv.convLocalMaxUndo(input, ingrad, output, outgrad, pool_size, start, stride, output_y,
            output_x, input_y)
        driver.Context.synchronize()

      elapsed = (time.time() - s) / self.count
      times.append(elapsed)
    
    if self.num_test == 0:
      return 100.0
    if self.num_test != 1:
      assert self.type == 'max'
      return max(times)
    return times[0]
      

class RNormExecuter(Executer):
  def execute(self):
    times = []
    for param in self.param:
      input_shape = tuple(param['input_shape'])
      output_shape = tuple(param['input_shape'])
 
      if input_shape[0] % 16 != 0:
        num_filter = (input_shape[0] + 16 -1 ) / 16 * 16
        input_shape = (num_filter, ) + input_shape[1:]
        output_shape = (num_filter, ) + output_shape[1:]
        print 'Change the number of filters to %d and  make it a multiple of 16' % num_filter
     
      input = gpuarray.to_gpu(np.random.randn(*input_shape).astype(np.float32))
      outgrad = gpuarray.to_gpu(np.random.randn(*input_shape).astype(np.float32))
      output = gpuarray.to_gpu(np.random.randn(*output_shape).astype(np.float32))
      ingrad = gpuarray.to_gpu(np.random.randn(*output_shape).astype(np.float32))
      
      denom = gpuarray.to_gpu(np.random.randn(*output_shape).astype(np.float32))

      input_y = input_shape[1]
      input_x = input_shape[2]
      output_y = output_shape[1]
      output_x = output_shape[2]

      channel = input_shape[0]
      size = param['size']
      scaler = param['scale']
      pow = param['pow']
      
      cudaconv.convResponseNorm(input, denom, output, channel, size, input_y, scaler, pow)
      cudaconv.convResponseNormUndo(ingrad, denom, input, output, outgrad, channel, size, input_y,
          scaler, pow, 0.0, 1.0)
      driver.Context.synchronize()

      start = time.time()
      for i in range(self.count):
        cudaconv.convResponseNorm(input, denom, output, channel, size, input_y, scaler, pow)
        cudaconv.convResponseNormUndo(ingrad, denom, input, output, outgrad, channel, size, input_y,
            scaler, pow, 0.0, 1.0)
        driver.Context.synchronize()

      elapsed = (time.time() - start) / self.count
      times.append(elapsed)
    
    if self.num_test == 0:
      return 100.0
    if self.num_test != 1:
      assert self.type == 'max'
      return max(times)
    return times[0]
   
class CMRNormExecuter(Executer):
  def execute(self):
    times = []
    for param in self.param:
      input_shape = tuple(param['input_shape'])
      output_shape = tuple(param['input_shape'])

      if input_shape[0] % 16 != 0:
        num_filter = (input_shape[0] + 16 -1 ) / 16 * 16
        input_shape = (num_filter, ) + input_shape[1:]
        output_shape = (num_filter, ) + output_shape[1:]
        print 'Change the number of filters to %d and  make it a multiple of 16' % num_filter

      
      input = gpuarray.to_gpu(np.random.randn(*input_shape).astype(np.float32))
      outgrad = gpuarray.to_gpu(np.random.randn(*input_shape).astype(np.float32))
      output = gpuarray.to_gpu(np.random.randn(*output_shape).astype(np.float32))
      ingrad = gpuarray.to_gpu(np.random.randn(*output_shape).astype(np.float32))
      
      denom = gpuarray.to_gpu(np.random.randn(*output_shape).astype(np.float32))

      input_y = input_shape[1]
      input_x = input_shape[2]
      output_y = output_shape[1]
      output_x = output_shape[2]

      channel = input_shape[0]
      size = param['size']
      scaler = param['scale']
      pow = param['pow']
      
      cudaconv.convResponseNormCrossMap(input, denom, output, channel, size, input_y, scaler, pow, False)
      cudaconv.convResponseNormCrossMapUndo(ingrad, denom, input, output, outgrad, channel, size, input_y,
          scaler, pow, False, 0.0, 1.0)
      driver.Context.synchronize()
      
      start = time.time()
      for i in range(self.count):
        cudaconv.convResponseNormCrossMap(input, denom, output, channel, size, input_y, scaler, pow, False)
        cudaconv.convResponseNormCrossMapUndo(ingrad, denom, input, output, outgrad, channel, size, input_y,
            scaler, pow, False, 0.0, 1.0)
        driver.Context.synchronize()
      elapsed = (time.time() - start) / self.count
      times.append(elapsed)
    
    if self.num_test == 0:
      return 100.0
    if self.num_test != 1:
      assert self.type == 'max'
      return max(times)
    return times[0]
   
class FCExecuter(Executer):
  def execute(self):
    times = []
    for param in self.param:
      input_shape = tuple(param['input_shape'])
      output_shape = tuple(param['output_shape'])
      weight_shape = tuple(param['weight_shape'])
      
      input = gpuarray.to_gpu(np.random.randn(*input_shape).astype(np.float32))
      output = gpuarray.to_gpu(np.random.randn(*output_shape).astype(np.float32))
      weight = gpuarray.to_gpu(np.random.randn(*weight_shape).astype(np.float32))
       
      ingrad = gpuarray.to_gpu(np.random.randn(*output_shape).astype(np.float32))
      outgrad = gpuarray.to_gpu(np.random.randn(*input_shape).astype(np.float32))

      drop_out = param['drop_out']

      garray.matrixmult(weight, input, dest = output)
      if drop_out > 0.0:
        drop_mask = gpuarray.to_gpu(np.random.uniform(0, 1, output.size).astype(np.float32).reshape(output.shape))
        garray.bigger_than_scaler(drop_mask, drop_out)
        garray.copy_to(output * drop_mask, output)
        garray.copy_to(ingrad * drop_mask, ingrad)

      garray.matrixmult(garray.transpose(weight), ingrad, dest = outgrad)
      garray.matrixmult(ingrad, garray.transpose(input), dest = weight)

      start = time.time()
      for i in range(self.count):
        # forward
        garray.matrixmult(weight, input, dest = output)
        if drop_out > 0.0:
          drop_mask = gpuarray.to_gpu(np.random.uniform(0, 1, output.size).astype(np.float32).reshape(output.shape))
          garray.bigger_than_scaler(drop_mask, drop_out)
          garray.copy_to(output * drop_mask, output)
          # backward
          garray.copy_to(ingrad * drop_mask, ingrad)

        garray.matrixmult(garray.transpose(weight), ingrad, dest = outgrad)
        garray.matrixmult(ingrad, garray.transpose(input), dest = weight)

      elapsed = (time.time() - start) / self.count
      times.append(elapsed)

    if self.num_test == 0:
      return 100.0
    if self.num_test != 1:
      assert self.type == 'max'
      return max(times)
    return times[0]

class SoftmaxExecuter(Executer):
  def execute(self):
    times = []
    for param in self.param:
      input_shape = param['input_shape']
      output_shape = param['output_shape']

      input = gpuarray.to_gpu(np.random.randn(*input_shape).astype(np.float32))
      outgrad = gpuarray.to_gpu(np.random.randn(*input_shape).astype(np.float32))
      output = gpuarray.to_gpu(np.random.randn(*output_shape).astype(np.float32))
      ingrad = gpuarray.to_gpu(np.array([np.random.choice(output_shape[0]) for i in
        range(output_shape[1])]).astype(np.float32)).reshape(output_shape[1], 1)

      maximum = input.maxto(axis = 1)
      garray.copy_to(input-maximum, output)
      garray.iexp(output)
      sum = output.sumto(axis = 1)
      garray.copy_to(output/sum, output)
      
      garray.softmax_bprop(output, ingrad, outgrad)

      start = time.time()
      for i in range(self.count):
        maximum = input.maxto(axis = 1)
        garray.copy_to(input-maximum, output)
        garray.iexp(output)
        sum = output.sumto(axis = 1)
        garray.copy_to(output/sum, output)
        
        garray.softmax_bprop(output, ingrad, outgrad)
      elapsed = (time.time() - start) / self.count
      times.append(elapsed)

    if self.num_test == 0:
      return 100.0
    if self.num_test != 1:
      assert self.type == 'max'
      return max(times)
    return times[0]

class NeuronExecuter(Executer):
  def execute(self):
    times = []
    for param in self.param:
      input_shape = tuple(param['input_shape'])
      input_shape = (np.prod(input_shape[:-1]), input_shape[-1])
      output_shape = tuple(param['output_shape'])
      output_shape = (np.prod(output_shape[:-1]), output_shape[-1])
      
      input = gpuarray.to_gpu(np.random.randn(*input_shape).astype(np.float32))
      outgrad = gpuarray.to_gpu(np.random.randn(*input_shape).astype(np.float32))
      output = gpuarray.to_gpu(np.random.randn(*output_shape).astype(np.float32))
      ingrad = gpuarray.to_gpu(np.random.randn(*output_shape).astype(np.float32))
      
      garray.relu_activate(input, output, 0.0)
      garray.relu_compute_grad(ingrad, output, outgrad, 0.0)

      start = time.time()
      for i in range(self.count):
        garray.relu_activate(input, output, 0.0)
        garray.relu_compute_grad(ingrad, output, outgrad, 0.0)

      elapsed = (time.time() - start) / self.count
      times.append(elapsed)

    if self.num_test == 0:
      return 100.0
    if self.num_test != 1:
      assert self.type == 'max'
      return max(times)
    return times[0]

register_executer('conv', ConvExecuter)
register_executer('neuron', NeuronExecuter)
register_executer('pool', PoolExecuter)
register_executer('rnorm', RNormExecuter)
register_executer('cmrnorm', CMRNormExecuter)
register_executer('fc', FCExecuter)
register_executer('softmax', SoftmaxExecuter)

class RequestExecuter(object):
  def __init__(self, filename, output_filename):
    self.filename = filename
    self.output_filename = output_filename
    self.comput_cost = {}
    self.open_request()
    
  def open_request(self):
    with open(self.filename) as f:
      self.requests = json.load(f)

  def execute(self):
    for request in self.requests:
      if 'end' in request:
        self.write_cost()
      else:
        print 'Running request ...'
        print request
        decr = request['decr']
        param = request['param']
        layer_name = decr['layer_name']
        num_worker = decr['workers']
        state = tuple(decr['state'])
        
        executer = get_executer(decr['op'])(decr, param)
        elapsed = executer.execute()
        print 'elapsed = %f second' % elapsed
        
        if layer_name not in self.comput_cost:
          self.comput_cost[layer_name] = {}
        self.comput_cost[layer_name][(state, num_worker)] = elapsed
        if param and param[0].get('overlapping', 0) !=  0:
          self.comput_cost[layer_name]['overlapping'] = param[0]['overlapping']

  def write_cost(self):
    with open(self.output_filename, 'w') as f:
      pickle.dump(self.comput_cost, f)

if __name__ == '__main__':
  executer = RequestExecuter('TeslaC2075-2.imagenet.cfg-req', 'output')
  executer.execute()
