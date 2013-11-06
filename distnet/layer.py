from distnet.util import divup, print_matrix
from distnet.weights import WEIGHTS, to_gpu
from distnet import util
import garray
import numpy as np

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
    self.name = name
    self.type = type
    self.disable_bprop = disable_bprop
    
    self.output = None
    self.output_grad = None

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
    
  def init_output(self): 
    out_shape = self.get_output_shape()
    rows = int(np.prod(out_shape[:3]))
    cols = out_shape[3]
    #util.log('Allocating: %s ', out_shape)
    self.output = garray.zeros((rows, cols), dtype=np.float32)
    self.output_grad = garray.zeros((rows, cols), dtype=np.float32)

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
    self.batch_size = image_shape[-1]
  
  def attach(self, prev):
    assert False, 'Must be first layer!'
    
  def fprop(self, input, output, train=TRAIN):
    garray.copy_to(input, output)

    if PFout:
      print_matrix(output, self.name)
  
  def bprop(self, grad, input, output, outGrad):
    pass
  
  def get_output_shape(self):
    return tuple(list(self.image_shape[:3]) + [self.batch_size]) 
  

class WeightedLayer(Layer):
  def __init__(self, name, type, epsW, epsB, initW, initB, momW, momB, wc, weight, bias,
      weightIncr , biasIncr, disable_bprop=False):
    Layer.__init__(self, name, type, disable_bprop)
    self.initW = initW
    self.initB = initB
    
    self.weight = WEIGHTS.empty('weight.' + self.name, epsW, momW, wc)
    self.bias = WEIGHTS.empty('bias.' + self.name, epsB, momB, 0.0)
   
    if weight is not None:
      self.weight.set_weight(weight) 
    if weightIncr is not None:
      self.weight.set_incr(weightIncr)
    
    if bias is not None:
      self.bias.set_weight(bias) 
    if biasIncr is not None:
      self.bias.set_incr(to_gpu(biasIncr))
    
  def _init_weights(self, weight_shape, bias_shape):
    if self.initB is None:
      self.initB = 0.0
    
    if self.initW is None:
      self.initW = 1.0 / np.sqrt(np.prod(weight_shape))
     
    self.bias.shape = bias_shape
    self.weight.shape = weight_shape
     
    if self.weight.wt is None:
      self.weight.set_weight(to_gpu(col_randn(weight_shape, np.float32) * self.initW))

    if self.bias.wt is None:
      self.bias.set_weight(to_gpu((np.ones(bias_shape, dtype=np.float32) * self.initB)))
  
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

    WeightedLayer.__init__(self, name, 'conv', 
                           epsW, epsB, initW, initB, momW, momB, wc, weight,
                           bias, weightIncr, biasIncr, disable_bprop)
    
    util.log('numFilter:%s padding:%s stride:%s initW:%s initB:%s, w: %s, b: %s',
             self.numFilter, self.padding, self.stride, self.initW, self.initB, 
             self.weight, self.bias)

  def attach(self, prev_layer):
    image_shape = prev_layer.get_output_shape()
    self.numColor, self.img_size, _, self.batch_size = image_shape
    self.outputSize = 1 + divup(2 * self.padding + self.img_size - self.filterSize, self.stride)
    util.log_info('%s %s %s %s: %s', self.padding, self.img_size, self.filterSize, self.stride,
                  self.outputSize)
    self.modules = self.outputSize ** 2

    weight_shape = (self.filterSize * self.filterSize * self.numColor, self.numFilter)
    bias_shape = (self.numFilter, 1)
    
    self._init_weights(weight_shape, bias_shape)
    self.tmp = garray.zeros((self.numFilter, 
                               self.get_single_img_size() * self.batch_size / self.numFilter),
                                dtype=np.float32)

  def change_batch_size(self, batch_size):
    Layer.change_batch_size(self, batch_size)
    self.tmp = garray.zeros((self.numFilter, 
                               self.get_single_img_size() * self.batch_size / self.numFilter),
                                dtype=np.float32)

  def get_cross_width(self): 
    return self.filterSize - 1

  def get_single_img_size(self):
    return self.modules * self.numFilter

  def get_output_shape(self):
    return (self.numFilter, self.outputSize, self.outputSize, self.batch_size)


  def fprop(self, input, output, train=TRAIN):
    garray.convolution(input, self.weight.wt, output, self.img_size, self.outputSize,
        self.outputSize, -self.padding, self.stride, self.numColor, 1)

    garray.copy_to(output, self.tmp)
    #from garray.cuda_kernel import add_vec_to_rows
    #garray.add_vec_to_rows(self.tmp, self.bias.wt)
    #garray.copy_to(self.tmp, output)
    garray.copy_to(self.tmp + self.bias.wt, output)

    if PFout:
      print_matrix(output, self.name)

  def bprop(self, grad, input, output, outGrad):
    self.weight.grad.fill(0)
    self.bias.grad.fill(0)

    # bprop to next layer
    garray.bconvolution(input, grad, self.weight.wt, outGrad, self.img_size, self.img_size,
        self.outputSize, -self.padding, self.stride, self.numColor)
    
    # bprop weight
    garray.wconvolution(input, grad, self.weight.grad, self.img_size, self.outputSize,
        self.outputSize, self.filterSize, -self.padding, self.stride, self.numColor)
    
    # bprop bias
    garray.copy_to(grad, self.tmp)
    self.bias.set_grad(garray.sum(self.tmp, axis = 1))


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
    self.numColor, self.img_size, _, self.batch_size = image_shape
    self.outputSize = divup(self.img_size - self.poolSize - self.start, self.stride) + 1
  
  def get_output_shape(self):
    return (self.numColor, self.outputSize, self.outputSize, self.batch_size)

  def get_cross_width(self): 
    return self.poolSize - 1

  def fprop(self, input, output, train=TRAIN):
    garray.maxpool(input, output, self.numColor, self.poolSize, self.start, self.stride, self.img_size,
        self.outputSize, self.outputSize)
    if PFout:
      print_matrix(output, self.name)

  def bprop(self, grad, input, output, outGrad):
    garray.maxundo(input, grad, output, outGrad, self.poolSize,
        self.start, self.stride, self.outputSize)

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
    self.numColor, self.img_size, _, self.batch_size = image_shape
    self.outputSize = divup(self.img_size - self.poolSize - self.start, self.stride) + 1

  def get_output_shape(self):
    return (self.numColor, self.outputSize, self.outputSize, self.batch_size)

  def get_cross_width(self): return self.poolSize - 1

  def fprop(self, input, output, train=TRAIN):
    garray.avgpool(input, output, self.numColor, self.poolSize, self.start, self.stride,
        self.img_size, self.outputSize, self.outputSize)
    if PFout:
      print_matrix(output, self.name)

  def bprop(self, grad, input, output, outGrad):
    garray.avgundo(input, grad, outGrad, self.poolSize,
        self.start, self.stride, self.outputSize, self.img_size)

class ResponseNormLayer(Layer):
  def __init__(self, name, pow=0.75, size=9, scale=0.001, disable_bprop=False):
    Layer.__init__(self, name, 'rnorm', disable_bprop)
    self.pow = pow
    self.size = size
    self.scale = scale
    self.scaler = self.scale / self.size ** 2
    self.denom = None
    util.log("pow:%s size:%s scale:%s scaler:%s", self.pow, self.size, self.scale, self.scaler)

  def attach(self, prev):
    image_shape = prev.get_output_shape()
    self.numColor, self.img_size, _, self.batch_size = image_shape

    self.denom = garray.zeros((self.numColor * self.img_size * self.img_size, self.batch_size), dtype = np.float32)


  def get_output_shape(self):
    return (self.numColor, self.img_size, self.img_size, self.batch_size)

  def change_batch_size(self, batch_size):
    Layer.change_batch_size(self, batch_size)
    self.denom = garray.zeros((self.numColor * self.img_size * self.img_size, self.batch_size), dtype = np.float32)

  def fprop(self, input, output, train=TRAIN):
    garray.rnorm(input, self.denom, output, self.numColor, self.size, self.scaler,
        self.pow)
    if PFout:
      print_matrix(output, self.name)

  def get_cross_width(self): return self.size - 1

  def bprop(self, grad, input, output, outGrad):
    garray.rnormundo(grad, self.denom, input, output, outGrad, self.numColor,
        self.size, self.scaler, self.pow)


class CrossMapResponseNormLayer(ResponseNormLayer):
  def __init__(self, name, pow=0.75, size=9, scale=0.001, blocked=False, disable_bprop=
      False):
    ResponseNormLayer.__init__(self, name, pow, size, scale, disable_bprop)
    self.type = 'cmrnorm'
    self.scaler = self.scale / self.size
    self.blocked = blocked

    util.log("pow:%s size:%s, scale:%s scaler:%s", self.pow, self.size, self.scale, self.scaler)

  def get_cross_width(self): return self.size - 1

  def fprop(self, input, output, train=TRAIN):
    garray.rnormcrossmap(input, self.denom, output, self.numColor, self.size, self.scaler, self.pow, self.blocked)
    if PFout:
      print_matrix(output, self.name)

  def bprop(self, grad, input, output, outGrad):
    garray.rnormcrossmapundo(grad, self.denom, input, output, outGrad, self.numColor,
        self.size, self.scaler, self.pow, self.blocked)


class FCLayer(WeightedLayer):
  def __init__(self, name, n_out, epsW=0.001, epsB=0.002, initW=None, initB=None,
      momW=0.9, momB=0.9, wc=0.004, dropRate=0.0, weight=None, bias=None, weightIncr=None, 
      biasIncr=None, disable_bprop=False):
    self.outputSize = n_out
    self.dropRate = dropRate

    WeightedLayer.__init__(self, name, 'fc', epsW, epsB, initW, initB, momW, momB, wc, weight,
        bias, weightIncr, biasIncr, disable_bprop)
    util.log('outputSize:%s initW:%s initB:%s dropRate:%s w: %s, b: %s',
        self.outputSize, self.initW, self.initB, self.dropRate, self.weight, self.bias)

  def attach(self, prev):
    input_shape = prev.get_output_shape()
    self.inputSize = int(np.prod(input_shape[0:3]))
    self.batch_size = input_shape[3]
    weight_shape = (self.outputSize, self.inputSize)
    bias_shape = (self.outputSize, 1)
    self._init_weights(weight_shape, bias_shape)


  def get_input_size(self): 
    return self.inputSize

  def get_output_shape(self):
    return (self.outputSize, 1, 1, self.batch_size)

  def fprop(self, input, output, train=TRAIN):
    garray.copy_to(garray.dot(self.weight.wt, input), output)
    garray.copy_to(output + self.bias.wt, output)
    if train == TEST:
      if self.dropRate > 0.0:
        output *= (1.0 - self.dropRate)
    else:
      if self.dropRate > 0.0:
        self.dropMask = to_gpu(np.random.uniform(0, 1, output.size).astype(np.float32).reshape(output.shape))
        garray.bigger_than_scaler(self.dropMask, self.dropRate)
        garray.copy_to(output * self.dropMask, output)
    if PFout:
      print_matrix(output, self.name)



  def bprop(self, grad, input, output, outGrad):
    if self.dropRate > 0.0:
      garray.copy_to(grad * self.dropMask, grad)
   
    garray.copy_to(garray.transpose(garray.dot(garray.transpose(grad), self.weight.wt)), outGrad)
    
    self.weight.set_grad(garray.dot(grad, garray.transpose(input)))
    self.bias.set_grad(garray.sum(grad, axis = 1))


class SoftmaxLayer(Layer):
  def __init__(self, name, disable_bprop=False):
    Layer.__init__(self, name, "softmax", disable_bprop)
    self.batchCorrect = 0

  def attach(self, prev_layer):
    input_shape = prev_layer.get_output_shape()
    self.inputSize, self.batch_size = int(np.prod(input_shape[0:3])), input_shape[3]
    self.outputSize = self.inputSize
    self.inputShape = input_shape
    self.cost = garray.zeros((self.batch_size, 1), dtype=np.float32)

  def get_output_shape(self):
    return (self.outputSize, 1, 1, self.batch_size)

  def fprop(self, input, output, train=TRAIN):
    #column max reduce
    max = garray.max(input, axis = 0)
    garray.copy_to(input - max, output)
    garray.iexp(output)
    sum = garray.sum(output, axis = 0)
    garray.copy_to(output /sum, output)
    if PFout:
      print_matrix(output, self.name)

  def logreg_cost(self, label, output):
    if self.cost.shape[0] != self.batch_size:
      self.cost = garray.zeros((self.batch_size, 1), dtype=np.float32)
    #column max id
    maxid = garray.argmax(output, axis = 0)
    self.batchCorrect = int(garray.sum(label == maxid).get())
    garray.logreg_cost_col(output, label, self.cost)

  def logreg_cost_multiview(self, label, output, num_view):
    unit = self.batch_size / num_view
    if self.cost.shape[0] != unit:
      self.cost = garray.zeros((unit, 1), dtype = np.float32)
    maxid = garray.argmax(output, axis = 0)
    self.batchCorrect = garray.same_reduce_multiview(label, maxid, num_view)
    tmp = garray.zeros((output.shape[0], unit), dtype = np.float32)
    garray.partial_copy_to(output, tmp, 0, output.shape[0], 0, unit)
    garray.logreg_cost_col(tmp, label, self.cost)

  def bprop(self, label, input, output, outGrad):
    garray.softmax_bprop(output, label, outGrad)


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
    garray.relu_activate(input, output, self.e)

  def computeGrad(self, grad, output, outGrad):
    garray.relu_compute_grad(grad, output, outGrad, self.e)

  def dump(self):
    d = Neuron.dump(self)
    d['e'] = self.e
    return d

class TanhNeuron(Neuron):
  def __init__(self, a, b):
    Neuron.__init__(self, 'tanh')
    self.a, self.b = a, b

  def activate(self, input, output):
    garray.tanh_activate(input, output, self.a , self.b)

  def computeGrad(self, grad, output, outGrad):
    garray.tanh_compute_grad(grad, output, outGrad, self.a, self.b)

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
    self.numColor, self.img_size, _, self.batch_size = image_shape

  def get_cross_width(self): 
    return 0

  def get_output_shape(self):
    return (self.numColor, self.img_size, self.img_size, self.batch_size)

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
