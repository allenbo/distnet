from distbase import util
from distbase.util import divup, print_matrix
from distnet.weights import WEIGHTS
import os
import numpy as np
import garray
from garray import ConvDataLayout, FilterLayout, FCDataLayout, WeightLayout

from multigpu import convert_shape, allocate, arr, uniformed_array, rank, multi_gpu
from multigpu import get_state, get_weight_distribution, get_output_distribution, get_bias_distribution

PFout = False
PBout = False
TEST = 0
TRAIN = 1

def col_rand(shape, dtype):
  return np.require(np.random.rand(*shape), dtype=dtype, requirements='C')

def col_randn(shape, dtype):
  return np.require(np.random.randn(*shape), dtype=dtype, requirements='C')


class Layer(object):
  def __init__(self, name, type, disable_bprop=False):
    self.merge_neuron = False
    self.name = name
    self.type = type
    self.disable_bprop = disable_bprop

    self.output = None
    self.output_grad = None
    self.neuron = None
    self.state = get_state(self.name)

  def disable_bprop(self):
    self.disable_bprop = True

  def enable_bprop(self):
    self.disable_bprop = False

  def change_batch_size(self, batch_size):
    self.batch_size = batch_size

  def attach(self, prev):
    self.init_output()

  def update(self):
    pass

  def init_output(self, fc = False):
    slice_dim = get_output_distribution(self.state, not fc)
    out_shape = self.get_output_shape()
    self.output = allocate(out_shape, slice_dim = slice_dim)
    self.output_grad = allocate(out_shape, slice_dim = slice_dim)

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
    Layer.__init__(self, name, 'data', True)
    self.name = name
    self.image_shape = image_shape
    self.batch_size = image_shape[ConvDataLayout.BATCH]

  def attach(self, prev):
    assert False, 'Must be first layer!'

  def fprop(self, input, output, train=TRAIN):
    arr.convert_from_data(input, output)

    if PFout:
      print_matrix(output, self.name)

  def bprop(self, grad, input, output, outGrad):
    pass

  def get_output_shape(self):
    image_shape = list(self.image_shape)
    image_shape[ConvDataLayout.BATCH] = self.batch_size
    return tuple(image_shape)


class WeightedLayer(Layer):
  def __init__(self, name, type, epsW, epsB, initW, initB, momW, momB, wc, weight, bias,
      weightIncr , biasIncr, disable_bprop=False):
    Layer.__init__(self, name, type, disable_bprop)
    self.initW = initW
    self.initB = initB

    slice_dim = get_weight_distribution(self.state, conv = (self.type == 'conv'))
    self.weight = WEIGHTS.empty('weight.' + self.name, epsW, momW, wc, slice_dim = slice_dim)
    slice_dim = get_bias_distribution(self.state, conv = (self.type == 'conv'))
    self.bias = WEIGHTS.empty('bias.' + self.name, epsB, momB, 0.0, slice_dim = slice_dim)

    if weight is not None:
      self.weight.set_weight(weight)
    if weightIncr is not None:
      self.weight.set_incr(weightIncr)

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

  def update(self):
    if self.disable_bprop:
      return

    self.weight.update(self.batch_size)
    self.bias.update(self.batch_size)

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
  def __init__(self, name, num_filters, filter_shape, padding=2, stride=1, initW=None,
               initB=None, partialSum=0, sharedBiases=0, epsW=0.001, epsB=0.002, momW=0.9, momB=0.9, wc=0.004,
      bias=None, weight=None, weightIncr=None, biasIncr=None, disable_bprop=False):

    self.numFilter = num_filters
    assert filter_shape[0] == filter_shape[1], 'Non-square filters not yet supported.'
    self.filterSize = filter_shape[0]
    self.padding = padding
    self.stride = stride

    self.partialSum = partialSum
    self.sharedBiases = sharedBiases

    if weight is not None:
      if len(weight.shape) == 2:
        num_filter = weight.shape[FilterLayout.NUM]
        num_color = weight.shape[FilterLayout.CHANNEL]
        new_shape = convert_shape(FilterLayout.get_filter_shape(self.filterSize, num_color, num_filter))
        weight = weight.reshape(new_shape)
        weightIncr = weightIncr.reshape(new_shape)

    WeightedLayer.__init__(self, name, 'conv',
                           epsW, epsB, initW, initB, momW, momB, wc, weight,
                           bias, weightIncr, biasIncr, disable_bprop)

    util.log('numFilter:%s padding:%s stride:%s initW:%s initB:%s, w: %s, b: %s',
             self.numFilter, self.padding, self.stride, self.initW, self.initB,
             self.weight, self.bias)
    self.merge_neuron = True

  def attach(self, prev_layer):
    image_shape = prev_layer.get_output_shape()
    self.numColor = image_shape[ConvDataLayout.CHANNEL]
    self.img_size = image_shape[ConvDataLayout.HEIGHT]
    self.batch_size = image_shape[ConvDataLayout.BATCH]
    self.outputSize = 1 + divup(2 * self.padding + self.img_size - self.filterSize, self.stride)
    util.log_info('%s %s %s %s: %s', self.padding, self.img_size, self.filterSize, self.stride,
                  self.outputSize)
    self.modules = self.outputSize ** 2

    weight_shape = FilterLayout.get_filter_shape(self.filterSize, self.numColor, self.numFilter)
    bias_shape = (self.numFilter, 1)

    self._init_weights(weight_shape, bias_shape)

  def get_output_shape(self):
    return ConvDataLayout.get_output_shape(self.outputSize, self.outputSize, self.numFilter, self.batch_size)


  def fprop(self, input, output, train=TRAIN):
    # Apply convolution operation to output
    # Merge bias addition to convolution operation
    arr.convolution(input, self.weight.wt, output, self.bias.wt, self.img_size, self.outputSize,
        self.outputSize, -self.padding, self.stride, self.numColor, 1)

    # Add bias to output
    #output.add(self.bias.wt, dst = output, shape = self.get_output_shape(), axis = ConvDataLayout.CHANNEL)
    if self.neuron == 'relu':
      arr.relu_activate(output, output, 0)

    if PFout:
      print_matrix(output, self.name)


  def bprop(self, grad, input, output, outGrad):
    self.weight.grad.fill(0)
    self.bias.grad.fill(0)

    if self.neuron == 'relu':
      arr.relu_compute_grad(grad, output, grad, 0)
    # bprop to next layer
    arr.bconvolution(input, grad, self.weight.wt, outGrad, self.img_size, self.img_size,
        self.outputSize, -self.padding, self.stride, self.numColor)

    # bprop weight
    arr.wconvolution(input, grad, self.weight.grad, self.bias.grad, self.img_size, self.outputSize,
        self.outputSize, self.filterSize, -self.padding, self.stride, self.numColor)

class MaxPoolLayer(Layer):
  def __init__(self, name, poolSize=2, stride=2, start=0, disable_bprop=False):
    Layer.__init__(self, name, 'pool', disable_bprop)
    self.pool = 'max'
    self.poolSize = poolSize
    self.stride = stride
    self.start = start
    util.log("pool_size:%s stride:%s start:%s", self.poolSize, self.stride, self.start)

  def attach(self, prev):
    image_shape = prev.get_output_shape()
    self.numColor = image_shape[ConvDataLayout.CHANNEL]
    self.img_size = image_shape[ConvDataLayout.HEIGHT]
    self.batch_size = image_shape[ConvDataLayout.BATCH]
    self.outputSize = divup(self.img_size - self.poolSize - self.start, self.stride) + 1

  def get_output_shape(self):
    return ConvDataLayout.get_output_shape(self.outputSize, self.outputSize, self.numColor, self.batch_size)

  def fprop(self, input, output, train=TRAIN):
    arr.maxpool(input, output, self.numColor, self.poolSize, self.start, self.stride, self.img_size,
        self.outputSize, self.outputSize)
    if PFout:
      print_matrix(output, self.name)

  def bprop(self, grad, input, output, outGrad):
    arr.maxundo(input, grad, output, outGrad, self.poolSize,
        self.start, self.stride, self.outputSize, self.outputSize, self.img_size)

class AvgPoolLayer(Layer):
  def __init__(self, name, poolSize=2, stride=2, start=0, disable_bprop=False):
    Layer.__init__(self, name, 'pool', disable_bprop)
    self.pool = 'avg'
    self.poolSize = poolSize
    self.stride = stride
    self.start = start
    util.log("pool_size:%s stride:%s start:%s", self.poolSize, self.stride, self.start)

  def attach(self, prev):
    image_shape = prev.get_output_shape()
    self.numColor = image_shpae[ConvDataLayout.CHANNEL]
    self.img_size = image_shape[ConvDataLayout.HEIGHT]
    self.batch_size = image_shape[ConvDataLayout.BATCH]
    self.outputSize = divup(self.img_size - self.poolSize - self.start, self.stride) + 1

  def get_output_shape(self):
    return ConvDataLayout.get_output_shape(self.outputSize, self.outputSize, self.numColor, self.batch_size)


  def fprop(self, input, output, train=TRAIN):
    arr.avgpool(input, output, self.numColor, self.poolSize, self.start, self.stride,
        self.img_size, self.outputSize, self.outputSize)
    if PFout:
      print_matrix(output, self.name)

  def bprop(self, grad, input, output, outGrad):
    arr.avgundo(input, grad, outGrad, self.poolSize,
        self.start, self.stride, self.outputSize, self.outputSize, self.img_size, self.img_size)

class ResponseNormLayer(Layer):
  def __init__(self, name, pow=0.75, size=9, scale=0.001, disable_bprop=False):
    Layer.__init__(self, name, 'rnorm', disable_bprop)
    self.pow = pow
    self.size = size
    self.scale = scale
    self.scalar = self.scale / self.size ** 2
    self.denom = None
    util.log("pow:%s size:%s scale:%s scalar:%s", self.pow, self.size, self.scale, self.scalar)

  def attach(self, prev):
    image_shape = prev.get_output_shape()
    self.numColor = image_shape[ConvDataLayout.CHANNEL]
    self.img_size  = image_shape[ConvDataLayout.HEIGHT]
    self.batch_size = image_shape[ConvDataLayout.BATCH]

    slice_dim = get_output_distribution(self.state, True)
    self.denom = allocate(image_shape, slice_dim = slice_dim)


  def get_output_shape(self):
    return ConvDataLayout.get_output_shape(self.img_size, self.img_size, self.numColor, self.batch_size)

  def change_batch_size(self, batch_size):
    Layer.change_batch_size(self, batch_size)
    slice_dim = get_output_distribution(self.state, True)
    self.denom = allocate(ConvDataLayout.get_output_shape(self.img_size , self.img_size, self.numColor, self.batch_size), slice_dim = slice_dim)

  def fprop(self, input, output, train=TRAIN):
    arr.rnorm(input, self.denom, output, self.numColor, self.size, self.img_size, self.scalar,
        self.pow)
    if PFout:
      print_matrix(output, self.name)


  def bprop(self, grad, input, output, outGrad):
    arr.rnormundo(grad, self.denom, input, output, outGrad, self.numColor,
        self.size, self.img_size, self.scalar, self.pow)


class CrossMapResponseNormLayer(ResponseNormLayer):
  def __init__(self, name, pow=0.75, size=9, scale=0.001, blocked=False, disable_bprop=
      False):
    ResponseNormLayer.__init__(self, name, pow, size, scale, disable_bprop)
    self.type = 'cmrnorm'
    self.scalar = self.scale / self.size
    self.blocked = blocked

    util.log("pow:%s size:%s, scale:%s scalar:%s", self.pow, self.size, self.scale, self.scalar)


  def fprop(self, input, output, train=TRAIN):
    arr.rnormcrossmap(input, self.denom, output, self.numColor, self.size, self.img_size, self.scalar, self.pow, self.blocked)
    if PFout:
      print_matrix(output, self.name)

  def bprop(self, grad, input, output, outGrad):
    arr.rnormcrossmapundo(grad, self.denom, input, output, outGrad, self.numColor,
        self.size, self.img_size,self.scalar, self.pow, self.blocked)


class FCLayer(WeightedLayer):
  ''' When the backend is caffe, we have to transpose the input to make batch as the second
  dimension of matrix'''
  def __init__(self, name, n_out, epsW=0.001, epsB=0.002, initW=None, initB=None,
      momW=0.9, momB=0.9, wc=0.004, dropRate=0.0, weight=None, bias=None, weightIncr=None,
      biasIncr=None, disable_bprop=False):
    self.outputSize = n_out
    self.dropRate = dropRate

    WeightedLayer.__init__(self, name, 'fc', epsW, epsB, initW, initB, momW, momB, wc, weight,
        bias, weightIncr, biasIncr, disable_bprop)
    util.log('outputSize:%s initW:%s initB:%s dropRate:%s w: %s, b: %s',
        self.outputSize, self.initW, self.initB, self.dropRate, self.weight, self.bias)
    self.merge_neuron = True
    self.prev_conv = False

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
    if self.prev_conv:
      self.input = arr.convert_to_fc(input)
    else:
      self.input = input
    
    arr.matrixmult(self.weight.wt, self.input,  dest = output)
    arr.copy_to(output + self.bias.wt, output)
    #output += self.bias.wt #output.add(self.bias.wt, dst = output, axis = 0)
    if train == TEST:
      if self.dropRate > 0.0:
        output *= (1.0 - self.dropRate)
    else:
      if self.dropRate > 0.0:
        self.dropMask = uniformed_array(np.random.uniform(0, 1, output.size).astype(np.float32).reshape(output.shape), slice_dim = None)
        arr.bigger_than_scalar(self.dropMask, self.dropRate)
        arr.copy_to(output * self.dropMask, output)

    if self.neuron == 'relu':
      arr.relu_activate(self.output, self.output, 0)
    if PFout:
      print_matrix(output, self.name)


  def bprop(self, grad, input, output, outGrad):
    if self.neuron == 'relu':
      arr.relu_compute_grad(grad, output, grad, 0)
    if self.dropRate > 0.0:
      arr.copy_to(grad * self.dropMask, grad)

    tmp = arr.matrixmult(arr.transpose(self.weight.wt), grad, dest = outGrad)
    if tmp != outGrad:
      arr.copy_to(tmp, outGrad)
    
    arr.matrixmult(grad, arr.transpose(self.input), dest = self.weight.grad)
    self.bias.set_grad(arr.sum(grad, axis = 1))

    if self.prev_conv:
      arr.copy_to(arr.convert_to_conv(grad), grad)


class SoftmaxLayer(Layer):
  def __init__(self, name, disable_bprop=False):
    Layer.__init__(self, name, "softmax", disable_bprop)
    self.batchCorrect = 0

  def attach(self, prev_layer):
    input_shape = prev_layer.get_output_shape()
    self.inputSize, self.batch_size = int(np.prod(input_shape[:-1])), input_shape[-1]
    self.outputSize = self.inputSize
    self.inputShape = input_shape
    self.create_cost(self.batch_size)

  def create_cost(self, size):
    self.cost = allocate((size, 1), slice_dim = None)

  def get_output_shape(self):
    return (self.outputSize, self.batch_size)

  def fprop(self, input, output, train=TRAIN):
    max = garray.max(input, axis = 0)
    arr.copy_to(input - max, output)
    arr.iexp(output)
    sum = arr.sum(output, axis = 0)
    arr.copy_to(output / sum, output)

    if PFout:
      print_matrix(output, self.name)


  def change_batch_size(self, batch_size):
    Layer.change_batch_size(self, batch_size)
    self.create_cost(self.batch_size)

  def logreg_cost(self, label, output):
    maxid = garray.argmax(output, axis = 0)
    self.batchCorrect = arr.sum(label == maxid)
    assert np.isscalar(self.batchCorrect)
    arr.logreg_cost_col(output, label, self.cost)

  def logreg_cost_multiview(self, label, output, num_view):
    # only try multiview with test on single gpu
    unit = self.batch_size / num_view
    if self.cost.shape[0] != unit:
      self.cost = allocate((unit, 1), dtype = np.float32, slice_dim = None)
    maxid = garray.argmax(output, axis = 0)
    self.batchCorrect = garray.same_reduce_multiview(label, maxid, num_view)
    tmp = allocate((output.shape[0], unit), dtype = np.float32, slice_dim = None)
    garray.partial_copy_to(output, tmp, 0, output.shape[0], 0, unit)
    garray.logreg_cost_col(tmp, label, self.cost)

  def bprop(self, label, input, output, outGrad):
    arr.softmax_bprop(output, label, outGrad)


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
  def __init__(self, name, type='relu', a=1.0, b=1.0, e=0.0, disable_bprop=False):
    Layer.__init__(self, name, 'neuron', disable_bprop)
    if type == 'relu':
      self.neuron = ReluNeuron(e)
    elif type == 'tanh':
      self.neuron = TanhNeuron(a, b)

  def attach(self, prev):
    image_shape = prev.get_output_shape()
    self.output_shape = image_shape
    if len(image_shape) == 4:
      self.numColor, self.img_size, _, self.batch_size = image_shape
    else:
      self.numColor, self.batch_size = image_shape
      self.img_size = 1

  def change_batch_size(self, batch_size):
    self.output_shape = tuple(list(self.output_shape)[:-1] + [batch_size])


  def get_output_shape(self):
    return self.output_shape

  def fprop(self, input, output, train=TRAIN):
    self.neuron.activate(input, output)
    if PFout:
      print_matrix(output, self.name)

  def bprop(self, grad, input, output, outGrad):
    self.neuron.computeGrad(grad, output, outGrad)

  def dump(self):
    d = Layer.dump(self)
    for k, v in self.neuron.dump().items():
      d[k] = v
    return d
