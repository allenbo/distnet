from distbase.state import sisw, disw_i
from distbase.util import issquare

import cudaconv
import cudaconv3
import caffe
import json
from pycuda import gpuarray, driver, autoinit
import numpy as np
import time
import garray
import cPickle as pickle

from mtime import MTime

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
    print '\033[32m%s %s\033[0m' % (self.decr['layer_name'], self.decr['state'])

  def execute(self):
    assert False, 'Implementation needed'

class ConvExecuter(Executer):
  def execute(self):
    times = []
    for param in self.param:
      print param
      input_shape = tuple(param['input_shape'])
      output_shape = tuple(param['output_shape'])
      filter_shape = tuple(param['filter_shape'])
      backend = param['backend']
      weight_sum = param['weight_sum']

      if backend == 'cudaconv':
        import cudaconv_backend as cm_backend
        operation = cudaconv
        if weight_sum == -1: weight_sum = 0
      elif backend == 'caffe':
        import caffe_backend as cm_backend
        operation = caffe
      elif backend == 'cudaconv3':
        import cudaconv3_backend as cm_backend
        operation = cudaconv3
        if weight_sum == -1: weight_sum = 10000
      else:
        assert False, 'There is no such backend %s' % (backend)


      channel_idx = cm_backend.ConvDataLayout.CHANNEL
      height_idx = cm_backend.ConvDataLayout.HEIGHT
      width_idx = cm_backend.ConvDataLayout.WIDTH
      filter_channel_idx = cm_backend.FilterLayout.CHANNEL
      filter_height_idx = cm_backend.FilterLayout.HEIGHT
      filter_num_idx = cm_backend.FilterLayout.NUM

      image_y = input_shape[height_idx]
      image_x = input_shape[width_idx]
      output_y = output_shape[height_idx]
      output_x = output_shape[width_idx]

      channel = filter_shape[filter_channel_idx]
      filter_size = filter_shape[filter_height_idx]
      num_filter = filter_shape[filter_num_idx]

      if filter_shape[filter_num_idx] % 16 != 0 and backend == 'cudaconv':
        num_filter = (filter_shape[filter_num_idx] + 16 - 1) / 16 * 16
        filter_shape = cm_backend.get_filter_shape(channel, filter_size, num_filter)
        print '\033[33mChange the number of filters to %d and  make it a multiple of 16\033[0m' % num_filter
        output_shape = cm_backend.get_image_shape(num_filter, output_y, output_x, output_shape[cm_backend.ConvDataLayout.BATCH])

      input = gpuarray.to_gpu(np.ndarray(input_shape).astype(np.float32))
      output = gpuarray.to_gpu(np.ndarray(output_shape).astype(np.float32))
      filter = gpuarray.to_gpu(np.ndarray(filter_shape).astype(np.float32))

      filter_grad = gpuarray.to_gpu(np.ndarray(filter_shape).astype(np.float32))

      ingrad = gpuarray.to_gpu(np.ndarray(output_shape).astype(np.float32))
      outgrad = gpuarray.to_gpu(np.ndarray(input_shape).astype(np.float32))
      bias = gpuarray.to_gpu(np.ndarray((num_filter, 1)).astype(np.float32))

      padding = param['padding']
      stride = param['stride']

      operation.convFilterActs(input, filter, output, bias,-padding, stride)
      driver.Context.synchronize()
      operation.convImgActs(input, ingrad, filter, outgrad, -padding, stride)
      driver.Context.synchronize()
      operation.convWeightActs(input, ingrad, filter, bias, -padding, stride, weight_sum)
      driver.Context.synchronize()

      fprop = bprop = wprop = update = 0
      fstart = time.time()
      for i in range(self.count):
        operation.convFilterActs(input, filter, output, bias, -padding, stride)
        driver.Context.synchronize()
      fprop = (time.time() - fstart) / self.count

      bstart = time.time()
      for i in range(self.count):
        operation.convImgActs(input, ingrad, filter, outgrad, -padding, stride)
        driver.Context.synchronize()
      bprop = (time.time() - bstart) / self.count

      wstart = time.time()
      for i in range(self.count):
        operation.convWeightActs(input, ingrad, filter, bias, -padding, stride, weight_sum)
        driver.Context.synchronize()
      wprop = (time.time() - wstart) / self.count

      ustart = time.time()
      for i in range(self.count):
        garray.matrix_add(filter_grad, filter)
        driver.Context.synchronize()
        garray.matrix_add(filter_grad, filter)
        driver.Context.synchronize()
        garray.matrix_add(filter_grad, filter)
        driver.Context.synchronize()
      update = (time.time() - ustart) / self.count

      times.append(MTime(fprop, bprop, wprop, update))

    if self.num_test == 0:
      return MTime(1000, 0)
    if self.num_test != 1:
      assert self.type == 'max'
      return max(times)
    return times[0]


class PoolExecuter(Executer):
  def execute(self):
    times = []
    for param in self.param:
      print param
      input_shape = tuple(param['input_shape'])
      output_shape = tuple(param['output_shape'])
      backend = param['backend']

      if backend == 'cudaconv':
        import cudaconv_backend as cm_backend
        operation = cudaconv
      elif backend == 'cudaconv3':
        import cudaconv3_backend as cm_backend
        operation = cudaconv3
      elif backend == 'caffe':
        import caffe_backend as cm_backend
        operation = caffe
      else:
        assert False, 'There is no such backend %s' % (backend)


      channel_idx = cm_backend.ConvDataLayout.CHANNEL
      height_idx = cm_backend.ConvDataLayout.HEIGHT
      width_idx = cm_backend.ConvDataLayout.WIDTH
      batch_idx = cm_backend.ConvDataLayout.BATCH

      input_y = input_shape[height_idx]
      input_x = input_shape[width_idx]
      output_y = output_shape[height_idx]
      output_x = output_shape[width_idx]

      if input_shape[channel_idx] % 16 != 0 and backend == 'cudaconv':
        num_filter = (input_shape[channel_idx] + 16 -1 ) / 16 * 16
        input_shape = cm_backend.get_image_shape(num_filter, input_y, input_x, input_shape[batch_idx])
        output_shape = cm_backend.get_image_shape(num_filter, output_y, output_x, output_shape[batch_idx])
        print '\033[33mChange the number of filters to %d and  make it a multiple of 16\033[0m' % num_filter

      channel = input_shape[channel_idx]
      #input_shape = (input_shape[channel_idx], input_y, input_x, input_shape[batch_idx])
      #output_shape = (output_shape[channel_idx], output_y, output_x, output_shape[batch_idx])
      input = gpuarray.to_gpu(np.ndarray(input_shape).astype(np.float32))
      outgrad = gpuarray.to_gpu(np.ndarray(input_shape).astype(np.float32))
      output = gpuarray.to_gpu(np.ndarray(output_shape).astype(np.float32))
      ingrad = gpuarray.to_gpu(np.ndarray(output_shape).astype(np.float32))

      pool_size = param['pool_size']
      start = param['start']
      stride = param['stride']

      operation.convLocalMaxPool(input, output, pool_size, start, stride)
      driver.Context.synchronize()
      operation.convLocalMaxUndo(input, ingrad, output, outgrad, pool_size, start, stride)
      driver.Context.synchronize()

      fprop = bprop = 0
      fstart = time.time()
      for i in range(self.count):
        # forward
        operation.convLocalMaxPool(input, output, pool_size, start, stride)
        driver.Context.synchronize()
      fprop = (time.time() - fstart) / self.count

      bstart = time.time()
      for i in range(self.count):
        # backward
        operation.convLocalMaxUndo(input, ingrad, output, outgrad, pool_size, start, stride)
        driver.Context.synchronize()
      bprop = (time.time() - bstart) / self.count

      times.append(MTime(fprop, bprop))

    if self.num_test == 0:
      return MTime(1000, 0)
    if self.num_test != 1:
      assert self.type == 'max'
      return max(times)
    return times[0]


class RNormExecuter(Executer):
  def execute(self):
    times = []
    for param in self.param:
      print param
      input_shape = tuple(param['input_shape'])
      output_shape = tuple(param['input_shape'])
      backend = param['backend']

      if backend == 'cudaconv':
        import cudaconv_backend as cm_backend
        operation = cudaconv
      elif backend == 'caffe':
        import caffe_backend as cm_backend
        operation = caffe
      elif backend == 'cudaconv3':
        import cudaconv3_backend as cm_backend
        operation = cudaconv3
      else:
        assert False, 'There is no such backend %s' % (backend)


      channel_idx = cm_backend.ConvDataLayout.CHANNEL
      height_idx = cm_backend.ConvDataLayout.HEIGHT
      width_idx = cm_backend.ConvDataLayout.WIDTH
      batch_idx = cm_backend.ConvDataLayout.BATCH

      input_y = input_shape[height_idx]
      input_x = input_shape[width_idx]
      output_y = output_shape[height_idx]
      output_x = output_shape[width_idx]


      if input_shape[channel_idx] % 16 != 0 and backend == 'cudaconv':
        num_filter = (input_shape[channel_idx] + 16 -1 ) / 16 * 16
        input_shape = cm_backend.get_image_shape(num_filter, input_y, input_x, input_shape[batch_idx])
        output_shape = cm_backend.get_image_shape(num_filter, output_y, output_x, output_shape[batch_idx])
        print '\033[33mChange the number of filters to %d and  make it a multiple of 16\033[0m' % num_filter

      input = gpuarray.to_gpu(np.ndarray(input_shape).astype(np.float32))
      outgrad = gpuarray.to_gpu(np.ndarray(input_shape).astype(np.float32))
      output = gpuarray.to_gpu(np.ndarray(output_shape).astype(np.float32))
      ingrad = gpuarray.to_gpu(np.ndarray(output_shape).astype(np.float32))

      denom = gpuarray.to_gpu(np.ndarray(output_shape).astype(np.float32))

      channel = input_shape[channel_idx]
      size = param['size']
      scalar = param['scale']
      pow = param['pow']

      operation.convResponseNorm(input, denom, output, size, scalar, pow)
      driver.Context.synchronize()
      operation.convResponseNormUndo(ingrad, denom, input, output, outgrad, size, scalar, pow)
      driver.Context.synchronize()

      fprop = bprop = 0
      fstart = time.time()
      for i in range(self.count):
        operation.convResponseNorm(input, denom, output, size, scalar, pow)
        driver.Context.synchronize()
      fprop = (time.time() - fstart) / self.count

      bstart = time.time()
      for i in range(self.count):
        operation.convResponseNormUndo(ingrad, denom, input, output, outgrad, size, scalar, pow)
        driver.Context.synchronize()
      bprop = (time.time() - bstart) / self.count

      times.append(MTime(fprop, bprop))

    if self.num_test == 0:
      return MTime(1000, 0)
    if self.num_test != 1:
      assert self.type == 'max'
      return max(times)
    return times[0]

class CMRNormExecuter(Executer):
  def execute(self):
    times = []
    for param in self.param:
      print param
      input_shape = tuple(param['input_shape'])
      output_shape = tuple(param['input_shape'])
      backend = param['backend']

      if backend == 'cudaconv':
        import cudaconv_backend as cm_backend
        operation = cudaconv
      elif backend == 'caffe':
        import caffe_backend as cm_backend
        operation = caffe
      elif backend == 'cudaconv3':
        import cudaconv3_backend as cm_backend
        operation = cudaconv3
      else:
        assert False, 'There is no such backend %s' % (backend)


      channel_idx = cm_backend.ConvDataLayout.CHANNEL
      height_idx = cm_backend.ConvDataLayout.HEIGHT
      width_idx = cm_backend.ConvDataLayout.WIDTH
      batch_idx = cm_backend.ConvDataLayout.BATCH

      input_y = input_shape[height_idx]
      input_x = input_shape[width_idx]
      output_y = output_shape[height_idx]
      output_x = output_shape[width_idx]


      if input_shape[channel_idx] % 16 != 0 and backend == 'cudaconv':
        num_filter = (input_shape[channel_idx] + 16 -1 ) / 16 * 16
        input_shape = cm_backend.get_image_shape(num_filter, input_y, input_x, input_shape[batch_idx])
        output_shape = cm_backend.get_image_shape(num_filter, output_y, output_x, output_shape[batch_idx])
        print '\033[33mChange the number of filters to %d and  make it a multiple of 16\033[0m' % num_filter

      channel = input_shape[channel_idx]
      #input_shape = (input_shape[channel_idx], input_y, input_x, input_shape[batch_idx])
      #output_shape = (output_shape[channel_idx], output_y, output_x, output_shape[batch_idx])
      input = gpuarray.to_gpu(np.ndarray(input_shape).astype(np.float32))
      outgrad = gpuarray.to_gpu(np.ndarray(input_shape).astype(np.float32))
      output = gpuarray.to_gpu(np.ndarray(output_shape).astype(np.float32))
      ingrad = gpuarray.to_gpu(np.ndarray(output_shape).astype(np.float32))

      denom = gpuarray.to_gpu(np.ndarray(output_shape).astype(np.float32))

      size = param['size']
      scalar = param['scale']
      pow = param['pow']

      operation.convResponseNormCrossMap(input, denom, output, size, scalar, pow, False)
      driver.Context.synchronize()
      operation.convResponseNormCrossMapUndo(ingrad, denom, input, output, outgrad, size, scalar, pow, False)
      driver.Context.synchronize()

      fprop = bprop = 0
      fstart = time.time()
      for i in range(self.count):
        operation.convResponseNormCrossMap(input, denom, output, size, scalar, pow, False)
        driver.Context.synchronize()
      fprop = (time.time() - fstart) / self.count

      bstart = time.time()
      for i in range(self.count):
        operation.convResponseNormCrossMapUndo(ingrad, denom, input, output, outgrad, size, scalar, pow, False)
        driver.Context.synchronize()
      bprop = (time.time() - bstart) / self.count
      times.append(MTime(fprop, bprop))

    if self.num_test == 0:
      return MTime(1000, 0)
    if self.num_test != 1:
      assert self.type == 'max'
      return max(times)
    return times[0]

class FCExecuter(Executer):
  def execute(self):
    times = []
    for param in self.param:
      print param
      input_shape = tuple(param['input_shape'])
      output_shape = tuple(param['output_shape'])
      weight_shape = tuple(param['weight_shape'])

      input = gpuarray.to_gpu(np.ndarray(input_shape).astype(np.float32))
      output = gpuarray.to_gpu(np.ndarray(output_shape).astype(np.float32))
      weight = gpuarray.to_gpu(np.ndarray(weight_shape).astype(np.float32))
      weight_grad = gpuarray.to_gpu(np.ndarray(weight_shape).astype(np.float32))

      ingrad = gpuarray.to_gpu(np.ndarray(output_shape).astype(np.float32))
      outgrad = gpuarray.to_gpu(np.ndarray(input_shape).astype(np.float32))

      time.sleep(1)
      drop_out = param['drop_out']

      garray.matrixmult(weight, input, dest = output)
      if drop_out > 0.0:
        drop_mask = gpuarray.to_gpu(np.random.uniform(0, 1, output.size).astype(np.float32).reshape(output.shape))
        garray.bigger_than_scalar(drop_mask, drop_out)
        garray.copy_to(output * drop_mask, output)
        garray.copy_to(ingrad * drop_mask, ingrad)

      garray.matrixmult(garray.transpose(weight), ingrad, dest = outgrad)
      garray.matrixmult(ingrad, garray.transpose(input), dest = weight)

      fprop = bprop = wprop = update = 0
      fstart = time.time()
      for i in range(self.count):
        # forward
        garray.matrixmult(weight, input, dest = output)
        if drop_out > 0.0:
          obj = np.random.uniform(0, 1, output.size).astype(np.float32).reshape(output.shape)
          drop_mask = gpuarray.to_gpu(obj)
          garray.bigger_than_scalar(drop_mask, drop_out)
          garray.copy_to(output * drop_mask, output)
      fprop = (time.time() - fstart) / self.count

      bstart = time.time()
      for i in range(self.count):
        # backward
        if drop_out > 0.0:
          garray.copy_to(ingrad * drop_mask, ingrad)

        garray.matrixmult(garray.transpose(weight), ingrad, dest = outgrad)
      bprop = (time.time() - bstart) / self.count

      wstart = time.time()
      for i in range(self.count):
        garray.matrixmult(ingrad, garray.transpose(input), dest = weight)
        driver.Context.synchronize()
      wprop = (time.time() - wstart) / self.count

      ustart = time.time()
      for i in range(self.count):
        garray.matrix_add(weight_grad, weight)
        driver.Context.synchronize()
        garray.matrix_add(weight_grad, weight)
        driver.Context.synchronize()
        garray.matrix_add(weight_grad, weight)
        driver.Context.synchronize()
      update = (time.time() - ustart) / self.count

      times.append(MTime(fprop, bprop, wprop, update))

    if self.num_test == 0:
      return MTime(1000, 0)
    if self.num_test != 1:
      assert self.type == 'max'
      return max(times)
    return times[0]

class SoftmaxExecuter(Executer):
  def execute(self):
    times = []
    for param in self.param:
      print param
      input_shape = param['input_shape']
      output_shape = param['output_shape']

      input = gpuarray.to_gpu(np.ndarray(input_shape).astype(np.float32))
      outgrad = gpuarray.to_gpu(np.ndarray(input_shape).astype(np.float32))
      output = gpuarray.to_gpu(np.ndarray(output_shape).astype(np.float32))
      ingrad = gpuarray.to_gpu(np.array([np.random.choice(output_shape[0]) for i in
        range(output_shape[1])]).astype(np.float32)).reshape(output_shape[1], 1)

      maximum = garray.max(input, axis = 0)
      garray.copy_to(input-maximum, output)
      garray.iexp(output)
      sum = garray.sum(output, axis = 0)
      garray.copy_to(output/sum, output)

      garray.softmax_bprop(output, ingrad, outgrad)

      fprop = bprop = 0
      fstart = time.time()
      for i in range(self.count):
        maximum = garray.max(input, axis = 0)
        garray.copy_to(input-maximum, output)
        garray.iexp(output)
        sum = garray.sum(output, axis = 0)
        garray.copy_to(output/sum, output)
      fprop = (time.time() - fstart) / self.count

      bstart = time.time()
      for i in range(self.count):
        garray.softmax_bprop(output, ingrad, outgrad)
      bprop = (time.time() - bstart) / self.count
      times.append(MTime(fprop, bprop))

    if self.num_test == 0:
      return MTime(1000, 0)
    if self.num_test != 1:
      assert self.type == 'max'
      return max(times)
    return times[0]

class NeuronExecuter(Executer):
  def execute(self):
    times = []
    for param in self.param:
      print param
      backend = param['backend']

      if backend == 'cudaconv':
        import cudaconv_backend as cm_backend
      elif backend == 'caffe':
        import caffe_backend as cm_backend
      elif backend == 'cudaconv3':
        import cudaconv3_backend as cm_backend
        operation = cudaconv3
      else:
        assert False, 'There is no such backend %s' % (backend)

      input_shape = tuple(param['input_shape'])

      if len(input_shape) == 2: # fc layer
        batch_idx = cm_backend.FCDataLayout.BATCH
      else:
        batch_idx = cm_backend.ConvDataLayout.BATCH

      input_shape = (np.prod(input_shape)/input_shape[batch_idx], input_shape[batch_idx])
      output_shape = tuple(param['output_shape'])
      output_shape = (np.prod(output_shape)/output_shape[batch_idx], output_shape[batch_idx])

      input = gpuarray.to_gpu(np.ndarray(input_shape).astype(np.float32))
      outgrad = gpuarray.to_gpu(np.ndarray(input_shape).astype(np.float32))
      output = gpuarray.to_gpu(np.ndarray(output_shape).astype(np.float32))
      ingrad = gpuarray.to_gpu(np.ndarray(output_shape).astype(np.float32))

      garray.relu_activate(input, output, 0.0)
      garray.relu_compute_grad(ingrad, output, outgrad, 0.0)

      fprop = bprop = 0
      fstart = time.time()
      for i in range(self.count):
        garray.relu_activate(input, output, 0.0)
      fprop = (time.time() - fstart) / self.count

      bstart = time.time()
      for i in range(self.count):
        garray.relu_compute_grad(ingrad, output, outgrad, 0.0)
      bprop = (time.time() - bstart) / self.count

      times.append(MTime(fprop, bprop))

    if self.num_test == 0:
      return MTime(1000, 0)
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
  def __init__(self, filename, output_filename, ideal = False):
    self.filename = filename
    self.output_filename = output_filename
    self.comput_cost = {}
    self.ideal = ideal
    self.open_request()

  def open_request(self):
    with open(self.filename) as f:
      self.requests = json.load(f)

  def execute(self):
    for request in self.requests:
      if 'end' in request:
        self.write_cost()
      else:
        decr = request['decr']
        param = request['param']
        layer_name = decr['layer_name']
        num_worker = decr['workers']
        state = tuple(decr['state'])

        print '\033[31mRunning request [%d workers] ...\033[0m' % (num_worker)

        if self.ideal and state != sisw: # assume the backend scale perfectly to any number of GPUs
          print 'Ideally scaled'
          elapsed = self.comput_cost[layer_name][sisw][0] / num_worker
        else:
          if state == disw_i and not issquare(num_worker):
            elapsed = MTime(1000, 0)
          else:
            executer = get_executer(decr['op'])(decr, param)
            elapsed = executer.execute()
        print 'elapsed = \033[1m%s\033[0m second' % elapsed

        if layer_name not in self.comput_cost:
          self.comput_cost[layer_name] = {}
        self.comput_cost[layer_name][state] = (elapsed, num_worker)
        if param and param[0].get('overlapping', 0) !=  0:
          self.comput_cost[layer_name]['overlapping'] = param[0]['overlapping']
        if param and param[0].get('actual_data', 0) !=  0:
          self.comput_cost[layer_name]['actual_data'] = param[0]['actual_data']

  def write_cost(self):
    with open(self.output_filename, 'w') as f:
      pickle.dump(self.comput_cost, f)

if __name__ == '__main__':
  executer = RequestExecuter('TeslaC2075-4.imagenet.cfg.cudaconv-req', 'output')
  executer.execute()
