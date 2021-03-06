from lr import ConstantLearningRate
from distbase import util
from distbase.monitor import MONITOR
from distbase.util import divup
from distnet.weights import WEIGHTS
import os
import sys
import numpy as np
import time
import garray
from garray import ConvDataLayout, FilterLayout, FCDataLayout, WeightLayout, backend_name

from multigpu import arr, multi_gpu
from distbase import state
from varray.para import FakePara

PFout = False
PBout = False
TEST = 0
TRAIN = 1
STOPITER = 1
OUTINDEX = [2]

def col_rand(shape, dtype):
  return np.require(np.random.rand(*shape), dtype=dtype, requirements='C')

def col_randn(shape, dtype):
  return np.require(np.random.randn(*shape), dtype=dtype, requirements='C')

class Layer(object):
  def __init__(self, name, type, para, disable_bprop=False):
    self.merge_neuron = False
    self.name = name
    self.type = type
    self._para = para
    self._prev_layer = None
    self._next_layer = None
    self.disable_bprop = disable_bprop

    self.output = None
    self.output_grad = None
    self.neuron = None
    self.iteration = 0

  def set_index(self, index):
    self.index = index

  def set_prev(self, layer):
    self._prev_layer = layer

  def set_next(self, layer):
    self._next_layer = layer

  def disable_bprop(self):
    self.disable_bprop = True

  def enable_bprop(self):
    self.disable_bprop = False

  def change_batch_size(self, batch_size):
    self.batch_size = batch_size

  def attach(self, prev):
    self.init_output()

  def update(self, stat):
    pass

  def prev_fprop(self):
    self.iteration += 1
    MONITOR.set_name(self.name)

  def prev_bprop(self):
    MONITOR.set_name(self.name)

  def _printout_forward(self, obj, fc = False):
    if PFout and self.index in OUTINDEX:
      obj.printout(self.name)
      if self.index == OUTINDEX[-1] and self.iteration == STOPITER:
        sys.exit(-1)

  def _printout_backward(self, obj_list, fc = False):
    if PBout and self.index in OUTINDEX:
      for obj in obj_list:
        if obj:
          obj.printout(self.name)
      if self.index == OUTINDEX[0] and self.iteration == STOPITER:
        sys.exit(-1)

  def init_output(self, fc = False):
    if self.type == 'data':
      self._para = self._next_layer._para
      output_allocate = self._para.init_input
    else:
      output_allocate = self._para.init_output


    out_shape = self.get_output_shape()
    self.output = output_allocate(out_shape)

    if self.type != 'data':
      self.output_grad = output_allocate(shape = out_shape)

  def dump(self):
    attr = [att for att in dir(self) if not att.startswith('__')]
    d = {}
    for att in attr:
      val = getattr(self, att)
      if isinstance(val, tuple) or np.isscalar(val):
        d[att] = val
    return d

class DataLayer(Layer):
  def __init__(self, name, image_shape):
    Layer.__init__(self, name, 'data', None, True)
    self.name = name
    self.image_shape = image_shape
    self.batch_size = image_shape[ConvDataLayout.BATCH]

  def attach(self, prev):
    assert False, 'Must be first layer!'

  def fprop(self, input, output, train=TRAIN):
    arr.convert_from_data(input, output)
    self._printout_forward(output)

  def bprop(self, grad, input, output, outGrad):
    pass

  def get_output_shape(self):
    image_shape = list(self.image_shape)
    image_shape[ConvDataLayout.BATCH] = self.batch_size
    return tuple(image_shape)

class WeightedLayer(Layer):
  def __init__(self, name, type, para, epsW, epsB, initW, initB, momW, momB, wc, weight, bias,
               weightIncr , biasIncr, disable_bprop=False, backend = 'cudaconv'):
    Layer.__init__(self, name, type, para, disable_bprop)
    self.initW = initW
    self.initB = initB

    self.weight = WEIGHTS.empty('weight.' + self.name, epsW, momW, wc, self._para)
    self.bias = WEIGHTS.empty('bias.' + self.name, epsB, momB, 0.0, self._para)

    if weight is not None:
      if backend == backend_name or self.type == 'fc':
        self.weight.set_weight(weight)
      else:
        self.weight.set_weight(arr.convert_from_backend(weight, backend))
    if weightIncr is not None:
      if backend == backend_name or self.type == 'fc':
        self.weight.set_incr(weightIncr)
      else:
        self.weight.set_incr(arr.convert_from_backend(weightIncr, backend))

    if bias is not None:
      self.bias.set_weight(bias)
    if biasIncr is not None:
      self.bias.set_incr(biasIncr)

  def _init_weights(self, weight_shape, bias_shape):
    if self.initB is None:
      self.initB = 0.0

    if self.initW is None:
      self.initW = 1.0 / np.sqrt(np.prod(weight_shape))

    self.bias.shape = bias_shape
    self.weight.shape = weight_shape

    if self.weight.wt is None:
      self.weight.set_weight(col_randn(weight_shape, np.float32) * self.initW)

    if self.bias.wt is None:
      self.bias.set_weight(np.ones(bias_shape, dtype=np.float32) * self.initB)

  def clear_weight_incr(self):
    self.weight.incr.fill(0)

  def clear_bias_incr(self):
    self.bias.incr.fill(0)

  def clear_incr(self):
    self.clear_weight_incr()
    self.clear_bias_incr()

  def update(self, stat):
    MONITOR.set_name(self.name)
    _ = time.time()
    if self.disable_bprop: return

    self.weight.update(stat)
    self.bias.update(stat)
    garray.driver.Context.synchronize()

    MONITOR.add_update(time.time() - _)

  def get_summary(self, type='mean'):
    w = self.weight.wt.get()
    w = np.mean(np.abs(w))
    w_variance = np.var(np.abs(w.ravel()))

    b = self.bias.wt.get()
    b = np.mean(np.abs(b))
    b_variance = np.var(np.abs(b.ravel()))
    return self.name, (w, w_variance, b, b_variance, self.weight.epsilon, self.bias.epsilon)

  def dump(self):
    d = Layer.dump(self)
    d['weight'] = self.weight.wt.get()
    d['bias'] = self.bias.wt.get()

    d['epsW'] = self.weight.epsilon
    d['momW'] = self.weight.momentum
    d['wc'] = self.weight.decay

    d['epsB'] = self.bias.epsilon
    d['momB'] = self.bias.momentum

    if self.weight.incr is not None:
      d['weightIncr'] = self.weight.incr.get()
    if self.bias.incr is not None:
      d['biasIncr'] = self.bias.incr.get()

    return d

class ConvLayer(WeightedLayer):
  def __init__(self, name, num_filters, filter_shape, para = FakePara(), padding=2, stride=1, initW=None,
               initB=None, partialSum=0, sumWidth=1000, sharedBiases=0, epsW=ConstantLearningRate(0.001),
               epsB=ConstantLearningRate(0.002), momW=0.9, momB=0.9, wc=0.004,
               bias=None, weight=None, weightIncr=None, biasIncr=None, disable_bprop=False, neuron =
               None, backend = 'cudaconv'):
    self.numFilter = num_filters
    assert filter_shape[0] == filter_shape[1], 'Non-square filters not yet supported.'
    self.filterSize = filter_shape[0]
    self.padding = padding
    self.stride = stride

    self.partialSum = partialSum
    self.sumWidth = sumWidth
    self.sharedBiases = sharedBiases

    WeightedLayer.__init__(self, name, 'conv', para,
                           epsW, epsB, initW, initB, momW, momB, wc, weight,
                           bias, weightIncr, biasIncr, disable_bprop, backend = backend)

    util.log('filter shape:%s padding:%s stride:%s initW:%s initB:%s, w: %s, b: %s',
             filter_shape, self.padding, self.stride, self.initW, self.initB, self.weight, self.bias)
    self.merge_neuron = True
    if neuron is not None:
      self.neuron = neuron
      util.log_info('Attach %s to %s', self.neuron, self.name)

  def attach(self, prev_layer):
    image_shape = prev_layer.get_output_shape()

    img_size = image_shape[ConvDataLayout.HEIGHT]
    self.numColor = image_shape[ConvDataLayout.CHANNEL]
    self.batch_size = image_shape[ConvDataLayout.BATCH]

    self.outputSize = 1 + divup(2 * self.padding + img_size - self.filterSize, self.stride)
    self.modules = self.outputSize ** 2

    weight_shape = FilterLayout.get_filter_shape(self.filterSize, self.numColor, self.numFilter)
    bias_shape = (self.numFilter, 1)
    self._init_weights(weight_shape, bias_shape)

  def get_output_shape(self):
    return ConvDataLayout.get_output_shape(self.outputSize, self.outputSize, self.numFilter, self.batch_size)

  def fprop(self, input, output, train=TRAIN):
    self._para.before_fprop(self)
    arr.convolution(input, self.weight.wt, output, self.bias.wt, -self.padding, self.stride)

    if self.neuron == 'relu':
      arr.relu_activate(output, output, 0)

    self._printout_forward(output)

  def bprop(self, grad, input, output, outGrad):
    self.weight.grad.fill(0)
    self.bias.grad.fill(0)

    if self.neuron == 'relu':
      arr.relu_compute_grad(grad, output, grad, 0)
    # bprop weight
    arr.wconvolution(input, grad, self.weight.grad, self.bias.grad, -self.padding, self.stride, self.sumWidth)
    self._para.after_weight(self)

    if self._prev_layer.type != 'data':
      # bprop to next layer
      arr.bconvolution(input, grad, self.weight.wt, outGrad, -self.padding, self.stride)
      self._para.after_bprop(self)

    self._printout_backward((self.bias.grad, self.weight.grad, outGrad))

class MaxPoolLayer(Layer):
  def __init__(self, name, para = FakePara(), poolSize=2, stride=2, start=0, disable_bprop=False):
    Layer.__init__(self, name, 'pool', para, disable_bprop)
    self.pool = 'max'
    self.poolSize = poolSize
    self.stride = stride
    self.start = start
    util.log("pool_size:%s stride:%s start:%s", self.poolSize, self.stride, self.start)

  def attach(self, prev):
    image_shape = prev.get_output_shape()
    img_size = image_shape[ConvDataLayout.HEIGHT]
    self.numColor = image_shape[ConvDataLayout.CHANNEL]
    self.batch_size = image_shape[ConvDataLayout.BATCH]
    self.outputSize = divup(img_size - self.poolSize - self.start, self.stride) + 1

  def get_output_shape(self):
    return ConvDataLayout.get_output_shape(self.outputSize, self.outputSize, self.numColor, self.batch_size)

  def fprop(self, input, output, train=TRAIN):
    self._para.before_fprop(self)
    arr.maxpool(input, output, self.poolSize, self.start, self.stride)
    self._printout_forward(output)

  def bprop(self, grad, input, output, outGrad):
    outGrad.fill(0)
    arr.maxundo(input, grad, output, outGrad, self.poolSize, self.start, self.stride)
    self._para.after_bprop(self)
    self._printout_backward((outGrad,))

class AvgPoolLayer(Layer):
  def __init__(self, name, para = FakePara(), poolSize=2, stride=2, start=0, disable_bprop=False):
    Layer.__init__(self, name, 'pool', para, disable_bprop)
    self.pool = 'avg'
    self.poolSize = poolSize
    self.stride = stride
    self.start = start
    util.log("pool_size:%s stride:%s start:%s", self.poolSize, self.stride, self.start)

  def attach(self, prev):
    image_shape = prev.get_output_shape()
    img_size = image_shape[ConvDataLayout.HEIGHT]
    self.numColor = image_shape[ConvDataLayout.CHANNEL]
    self.batch_size = image_shape[ConvDataLayout.BATCH]
    self.outputSize = divup(img_size - self.poolSize - self.start, self.stride) + 1

  def get_output_shape(self):
    return ConvDataLayout.get_output_shape(self.outputSize, self.outputSize, self.numColor, self.batch_size)

  def fprop(self, input, output, train=TRAIN):
    self._para.before_fprop(self)
    arr.avgpool(input, output,self.poolSize, self.start, self.stride)
    self._printout_forward(output)

  def bprop(self, grad, input, output, outGrad):
    arr.avgundo(input, grad, outGrad, self.poolSize, self.start, self.stride)
    self._para.after_bprop(self)
    self._printout_backward((outGrad,))

class ResponseNormLayer(Layer):
  def __init__(self, name, para = FakePara(), pow=0.75, size=9, scale=0.001, disable_bprop=False):
    Layer.__init__(self, name, 'rnorm', para,  disable_bprop)
    self.pow = pow
    self.size = size
    self.scale = scale
    self.scalar = self.scale / self.size ** 2
    self.denom = None
    util.log("pow:%s size:%s scale:%s scalar:%s", self.pow, self.size, self.scale, self.scalar)

  def attach(self, prev):
    image_shape = prev.get_output_shape()
    self.numColor = image_shape[ConvDataLayout.CHANNEL]
    self.outputSize  = image_shape[ConvDataLayout.HEIGHT]
    self.batch_size = image_shape[ConvDataLayout.BATCH]

  def get_output_shape(self):
    return ConvDataLayout.get_output_shape(self.outputSize, self.outputSize, self.numColor, self.batch_size)

  def change_batch_size(self, batch_size):
    Layer.change_batch_size(self, batch_size)
    self.denom = self._para.init_output(shape = self.get_output_shape())

  def fprop(self, input, output, train=TRAIN):
    self._para.before_fprop(self)
    arr.rnorm(input, self.denom, output, self.size, self.scalar, self.pow)
    self._printout_forward(output)

  def bprop(self, grad, input, output, outGrad):
    outGrad.fill(0)
    arr.rnormundo(grad, self.denom, input, output, outGrad, self.size, self.scalar, self.pow)
    self._para.after_bprop(self)
    self._printout_backward((outGrad,))

class CrossMapResponseNormLayer(ResponseNormLayer):
  def __init__(self, name, para = FakePara(), pow=0.75, size=9, scale=0.001, blocked=False, disable_bprop=
      False):
    ResponseNormLayer.__init__(self, name, para, pow, size, scale, disable_bprop)
    self.type = 'cmrnorm'
    self.scalar = self.scale / self.size
    self.blocked = blocked

    util.log("pow:%s size:%s, scale:%s scalar:%s", self.pow, self.size, self.scale, self.scalar)


  def fprop(self, input, output, train=TRAIN):
    self._para.before_fprop(self)
    arr.rnormcrossmap(input, self.denom, output, self.size, self.scalar, self.pow, self.blocked)
    self._printout_forward(output)

  def bprop(self, grad, input, output, outGrad):
    outGrad.fill(0)
    arr.rnormcrossmapundo(grad, self.denom, input, output, outGrad, self.size,self.scalar, self.pow, self.blocked)
    self._para.after_bprop(self)
    self._printout_backward((outGrad,))

class FCLayer(WeightedLayer):
  ''' When the backend is caffe, we have to transpose the input to make batch as the second
  dimension of matrix'''
  def __init__(self, name, n_out, para = FakePara(), epsW=ConstantLearningRate(0.001), epsB=ConstantLearningRate(0.002), initW=None, initB=None,
      momW=0.9, momB=0.9, wc=0.004, dropRate=0.0, weight=None, bias=None, weightIncr=None,
      biasIncr=None, disable_bprop=False, neuron = None):
    self.outputSize = n_out
    self.dropRate = dropRate

    WeightedLayer.__init__(self, name, 'fc', para, epsW, epsB, initW, initB, momW, momB, wc, weight,
        bias, weightIncr, biasIncr, disable_bprop)
    util.log('outputSize:%s initW:%s initB:%s dropRate:%s w: %s, b: %s',
        self.outputSize, self.initW, self.initB, self.dropRate, self.weight, self.bias)
    self.merge_neuron = True
    self.prev_conv = False
    if neuron is not None:
      self.neuron = neuron
      util.log_info('Attach %s to %s', self.neuron, self.name)

  def attach(self, prev):
    input_shape = prev.get_output_shape()
    if len(input_shape) == 4:
      self.batch_size = input_shape[ConvDataLayout.BATCH]
      self.prev_conv = True # previous layer is a conv-related layer, needs 4 dimension input and output
    else:
      self.batch_size = input_shape[FCDataLayout.BATCH]

    self.inputSize = int(np.prod(input_shape)) / self.batch_size
    weight_shape = WeightLayout.get_weight_shape(self.inputSize, self.outputSize)

    bias_shape = (self.outputSize, 1)
    self._init_weights(weight_shape, bias_shape)

  def get_input_size(self):
    return self.inputSize

  def get_output_shape(self):
    return FCDataLayout.get_output_shape(self.outputSize, self.batch_size)

  def fprop(self, input, output, train=TRAIN):
    self._para.before_fprop(self)
    arr.fcforward(input, output, self.weight.wt, self.bias.wt, self.prev_conv)
    if train == TEST:
      if self.dropRate > 0.0:
        output *= (1.0 - self.dropRate)
    else:
      if self.dropRate > 0.0:
        self.dropMask = self._para.random_uniform(shape = output.shape)
        arr.bigger_than_scalar(self.dropMask, self.dropRate)
        arr.copy_to(output * self.dropMask, output)

    if self.neuron == 'relu':
      arr.relu_activate(self.output, self.output, 0)
    self._printout_forward(output, fc = True)
    garray.driver.Context.synchronize()

  def bprop(self, grad, input, output, outGrad):
    outGrad.fill(0)
    self.weight.grad.fill(0)
    self.bias.grad.fill(0)

    if self.neuron == 'relu':
      arr.relu_compute_grad(grad, output, grad, 0)
    if self.dropRate > 0.0:
      arr.copy_to(grad * self.dropMask, grad)

    garray.driver.Context.synchronize()
    arr.fcbackward(input, self.weight.wt, grad, outGrad, self.weight.grad, self.bias.grad, self.prev_conv)
    self._para.after_bprop(self)
    self._para.after_weight(self)

    self._printout_backward((self.weight.grad, ), fc = True)

class SoftmaxLayer(Layer):
  def __init__(self, name, para = FakePara(), disable_bprop=False):
    # softmax layer has to apply replica parallel, or fake parallel when it's single GPU
    Layer.__init__(self, name, "softmax", para, disable_bprop)
    self.batchCorrect = 0
    assert self._para.name == 'R' or self._para.name == 'F'

  def attach(self, prev_layer):
    input_shape = prev_layer.get_output_shape()
    self.inputSize, self.batch_size = int(np.prod(input_shape[:-1])), input_shape[-1]
    self.outputSize = self.inputSize
    self.inputShape = input_shape
    self.create_cost(self.batch_size)

  def create_cost(self, size):
    if size < 0:
      return
    self.cost = self._para.init_output(shape = (1, size))

  def get_output_shape(self):
    return (self.outputSize, self.batch_size)

  def fprop(self, input, output, train=TRAIN):
    self._para.before_fprop(self)
    arr.softmax(input, output)

    self._printout_forward(output, fc = True)

  def change_batch_size(self, batch_size):
    Layer.change_batch_size(self, batch_size)
    self.create_cost(self.batch_size)

  def logreg_cost(self, label, output):
    maxid = arr.argmax(output, axis = 0)
    self.batchCorrect = arr.sum(maxid == label)
    assert np.isscalar(self.batchCorrect)
    arr.logreg_cost_col(output, label, self.cost)

  def bprop(self, label, input, output, outGrad):
    outGrad.fill(0)
    arr.softmax_bprop(output, label, outGrad)
    self._printout_backward((outGrad,), fc = True)

  def get_correct(self):
    return  1.0 * self.batchCorrect / self.batch_size


class Neuron:
  def __init__(self, type):
    self.type = type

  def activate(self, input, output):
    assert False, 'No Implementation of Activation'

  def computeGrad(self, grad, output, inputGrad):
    assert False, 'No Implementation of Gradient'

  def dump(self):
    return {'neuron': self.type}

class ReluNeuron(Neuron):
  def __init__(self, e):
    Neuron.__init__(self, 'relu')
    self.e = e;

  def activate(self, input, output):
    arr.relu_activate(input, output, self.e)

  def computeGrad(self, grad, output, outGrad):
    arr.relu_compute_grad(grad, output, outGrad, self.e)

  def dump(self):
    d = Neuron.dump(self)
    d['e'] = self.e
    return d

class TanhNeuron(Neuron):
  def __init__(self, a, b):
    Neuron.__init__(self, 'tanh')
    self.a, self.b = a, b

  def activate(self, input, output):
    arr.tanh_activate(input, output, self.a , self.b)

  def computeGrad(self, grad, output, outGrad):
    arr.tanh_compute_grad(grad, output, outGrad, self.a, self.b)

  def dump(self):
    d = Neuron.dump(self)
    d['a'] = self.a
    d['b'] = self.b
    return d

class NeuronLayer(Layer):
  def __init__(self, name, para = FakePara(), type='relu', a=1.0, b=1.0, e=0.0, disable_bprop=False):
    Layer.__init__(self, name, 'neuron', para,  disable_bprop)
    if type == 'relu':
      self.neuron = ReluNeuron(e)
    elif type == 'tanh':
      self.neuron = TanhNeuron(a, b)

  def attach(self, prev):
    image_shape = prev.get_output_shape()
    self.output_shape = image_shape
    if len(image_shape) == 4:
      self.numColor, self.img_size, _, self.batch_size = image_shape
      self._para == self._para.to_conv()
    else:
      self.numColor, self.batch_size = image_shape
      self.img_size = 1
      self._para == self._para.to_fc()

  def change_batch_size(self, batch_size):
    self.output_shape = tuple(list(self.output_shape)[:-1] + [batch_size])

  def get_output_shape(self):
    return self.output_shape

  def fprop(self, input, output, train=TRAIN):
    self.neuron.activate(input, output)
    self._printout_forward(output)

  def bprop(self, grad, input, output, outGrad):
    self.neuron.computeGrad(grad, output, outGrad)
    self._printout_backward((outGrad, ))

  def dump(self):
    d = Layer.dump(self)
    for k, v in self.neuron.dump().items():
      d[k] = v
    return d
