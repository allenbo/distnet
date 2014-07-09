from distnet import net
from distnet.layer import ConvLayer, MaxPoolLayer, AvgPoolLayer, \
  CrossMapResponseNormLayer, SoftmaxLayer, NeuronLayer, ResponseNormLayer, FCLayer, \
  DataLayer
import distnet
from distbase import util
from distbase.util import isfloat
from distbase.util import parse_config_file

import numpy as np

parse_config_file = parse_config_file

def load_model(net, model):
  if 'layers' in model:
    # Loading from a checkpoint
    add_layers(FastNetBuilder(), net, model['layers'])
  else:
    if is_cudaconvnet_config(model):
      # AlexK config file
      add_layers(CudaconvNetBuilder(), net, model)
    else:
      # FastNet config file
      add_layers(FastNetBuilder(), net, model)
      
  return net

def load_from_checkpoint(config, checkpoint, image_shape):
  network = net.FastNet(image_shape)
  if checkpoint is not None:
    load_model(network, checkpoint)
  else:
    load_model(network, parse_config_file(config))
  return network
    

class Builder(object):
  valid_dic = {}
  @staticmethod
  def set_val(ld, name, default = None):
    val  = ld.get(name, default)
    Builder.valid_dic[name] = 1
    return val

  @staticmethod
  def check_opts(ld):
    for k in Builder.valid_dic:
      if k not in ld:
        raise Exception, 'Unknown key %s' % k
    else:
      Builder.valid_dic = {}

  def make_layer(self, net, ld):
    if ld['type'] == 'conv': return self.conv_layer(ld)
    elif ld['type'] == 'pool': return self.pool_layer(ld)
    elif ld['type'] == 'neuron': return self.neuron_layer(ld)
    elif ld['type'] == 'fc': return self.fc_layer(ld)
    elif ld['type'] == 'softmax': return self.softmax_layer(ld)
    elif ld['type'] == 'rnorm': return self.rnorm_layer(ld)
    elif ld['type'] == 'cmrnorm': return self.crm_layer(ld)
    else:
      return None
      #raise Exception, 'Unknown layer %s' % ld['type']


class FastNetBuilder(Builder):
  def conv_layer(self, ld):
    numFilter = Builder.set_val(ld, 'numFilter')
    filterSize = Builder.set_val(ld, 'filterSize')
    padding = Builder.set_val(ld, 'padding')
    stride = Builder.set_val(ld, 'stride')
    initW = Builder.set_val(ld, 'initW', 0.01)
    initB = Builder.set_val(ld, 'initB', 0.00)
    epsW = Builder.set_val(ld, 'epsW', 0.001)
    epsB = Builder.set_val(ld, 'epsB', 0.002)
    if epsB == 0:
      epsB = 0.002
    momW = Builder.set_val(ld, 'momW', 0.0)
    momB = Builder.set_val(ld, 'momB', 0.0)
    sharedBiases = Builder.set_val(ld, 'sharedBiases', default = 1)
    partialSum = Builder.set_val(ld, 'partialSum', default = 0)
    wc = Builder.set_val(ld, 'wc', 0.0)
    bias = Builder.set_val(ld, 'bias')
    weight = Builder.set_val(ld, 'weight')
    weightIncr = Builder.set_val(ld, 'weightIncr')
    biasIncr = Builder.set_val(ld, 'biasIncr')
    name = Builder.set_val(ld, 'name')
    neuron = Builder.set_val(ld, 'neuron')
    disable_bprop = Builder.set_val(ld, 'disable_bprop', default = False)
    cv = ConvLayer(name, numFilter, (filterSize, filterSize), padding, stride, initW, initB,
        partialSum,sharedBiases, epsW, epsB, momW, momB, wc, bias, weight,
        weightIncr = weightIncr, biasIncr = biasIncr, disable_bprop = disable_bprop, neuron = neuron)
    return cv

  def pool_layer(self, ld):
    stride = Builder.set_val(ld, 'stride')
    start = Builder.set_val(ld, 'start')
    poolSize = Builder.set_val(ld, 'poolSize')
    name = Builder.set_val(ld, 'name')
    pool = Builder.set_val(ld, 'pool', default = 'max')
    disable_bprop = Builder.set_val(ld, 'disable_bprop', default = False)
    if pool == 'max':
      return MaxPoolLayer(name, poolSize, stride, start, disable_bprop = disable_bprop)
    elif pool == 'avg':
      return AvgPoolLayer(name, poolSize, stride, start, disable_bprop = disable_bprop)

  def crm_layer(self, ld):
    name = Builder.set_val(ld, 'name')
    pow = Builder.set_val(ld, 'pow')
    size = Builder.set_val(ld, 'size')
    scale = Builder.set_val(ld, 'scale')
    blocked = bool(Builder.set_val(ld, 'blocked', default = 0))
    disable_bprop = Builder.set_val(ld, 'disable_bprop', default = False)
    return CrossMapResponseNormLayer(name, pow, size, scale, blocked, disable_bprop =
        disable_bprop)

  def softmax_layer(self, ld):
    name = Builder.set_val(ld, 'name')
    disable_bprop = Builder.set_val(ld, 'disable_bprop', default = False)
    return SoftmaxLayer(name, disable_bprop = disable_bprop)

  def neuron_layer(self, ld):
    name = Builder.set_val(ld, 'name')
    disable_bprop = Builder.set_val(ld, 'disable_bprop', default = False)
    if ld['neuron'] == 'relu':
      e = Builder.set_val(ld, 'e')
      return NeuronLayer(name, type='relu', e=e, disable_bprop = disable_bprop)

    if ld['neuron'] == 'tanh':
      a = Builder.set_val(ld, 'a')
      b = Builder.set_val(ld, 'b')
      return NeuronLayer(name, type='tanh', a=a, b=b, disable_bprop = disable_bprop)

    assert False, 'No implementation for the neuron type' + ld['neuron']['type']

  def rnorm_layer(self, ld):
    name = Builder.set_val(ld, 'name')
    pow = Builder.set_val(ld,'pow')
    size = Builder.set_val(ld, 'size')
    scale = Builder.set_val(ld, 'scale')
    disable_bprop = Builder.set_val(ld, 'disable_bprop', default = False)
    return ResponseNormLayer(name, pow, size, scale, disable_bprop = disable_bprop)


  def fc_layer(self, ld):
    epsB = Builder.set_val(ld, 'epsB', 0.002)
    if epsB == 0:
      epsB = 0.002
    epsW = Builder.set_val(ld ,'epsW', 0.001)
    initB = Builder.set_val(ld, 'initB', 0.00)
    initW = Builder.set_val(ld, 'initW', 0.01)
    momB = Builder.set_val(ld, 'momB', 0.0)
    momW = Builder.set_val(ld, 'momW', 0.0)
    wc = Builder.set_val(ld, 'wc', 0.0)
    dropRate = Builder.set_val(ld, 'dropRate', 0.0)

    n_out = Builder.set_val(ld , 'outputSize')
    bias = Builder.set_val(ld, 'bias')
    weight = Builder.set_val(ld, 'weight')

    weightIncr = Builder.set_val(ld, 'weightIncr')
    biasIncr = Builder.set_val(ld, 'biasIncr')
    name = Builder.set_val(ld, 'name')
    neuron = Builder.set_val(ld, 'neuron')
    disable_bprop = Builder.set_val(ld, 'disable_bprop', default = False)
    return FCLayer(name, n_out, epsW, epsB, initW, initB, momW, momB, wc, dropRate,
        weight, bias, weightIncr = weightIncr, biasIncr = biasIncr, disable_bprop = disable_bprop, neuron = neuron)




class CudaconvNetBuilder(FastNetBuilder):
  def conv_layer(self, ld):
    numFilter = ld['filters']
    filterSize = ld['filterSize']
    padding = ld['padding']
    stride = ld['stride']
    initW = ld['initW']
    initB = ld.get('initB', 0.0)
    name = ld['name']
    epsW = ld['epsW']
    epsB = ld['epsB']

    momW = ld['momW']
    momB = ld['momB']

    wc = ld['wc']

    bias = ld.get('biases', None)
    weight = ld.get('weights', None)

    return ConvLayer(name, numFilter, (filterSize, filterSize), padding, stride, initW, initB, 0, 0, epsW, epsB, momW
        = momW, momB = momB, wc = wc, bias = bias, weight = weight)

  def pool_layer(self, ld):
    stride = ld['stride']
    start = ld['start']
    poolSize = ld['sizeX']
    name = ld['name']
    pool = ld['pool']
    if pool == 'max':
      return MaxPoolLayer(name, poolSize, stride, start)
    else:
      return AvgPoolLayer(name, poolSize, stride, start)


  def neuron_layer(self, ld):
    if ld['neuron']['type'] == 'relu':
      name = ld['name']
      #e = ld['neuron']['e']
      return NeuronLayer(name, type='relu')
    if ld['neuron']['type'] == 'tanh':
      name = ld['name']
      a = ld['neuron']['a']
      b = ld['neuron']['b']
      return NeuronLayer(name, 'tanh', a=a, b=b)

    assert False, 'No implementation for the neuron type' + ld['neuron']['type']


  def fc_layer(self, ld):
    epsB = ld['epsB']
    epsW = ld['epsW']
    initB = ld.get('initB', 0.0)
    initW = ld['initW']
    momB = ld['momB']
    momW = ld['momW']

    wc = ld['wc']
    dropRate = ld.get('dropRate', 0.0)

    n_out = ld['outputs']
    bias = ld.get('biases', None)
    weight = ld.get('weights', None)

    if bias is not None:
      bias = bias.transpose()
      bias = np.require(bias, dtype = np.float32, requirements = 'C')
    if weight is not None:
      weight = weight.transpose()
      weight = np.require(weight, dtype = np.float32, requirements = 'C')

    name = ld['name']
    return FCLayer(name, n_out, epsW, epsB, initW, initB, momW = momW, momB = momB, wc
        = wc, dropRate = dropRate, weight = weight, bias = bias)

  def rnorm_layer(self, ld):
    name = Builder.set_val(ld, 'name')
    pow = Builder.set_val(ld,'pow')
    size = Builder.set_val(ld, 'size')
    scale = Builder.set_val(ld, 'scale')
    #scale = scale * size ** 2
    return ResponseNormLayer(name, pow, size, scale)

  def crm_layer(self, ld):
    name = Builder.set_val(ld, 'name')
    pow = Builder.set_val(ld, 'pow')
    size = Builder.set_val(ld, 'size')
    scale = Builder.set_val(ld, 'scale')
    #scale = scale * size
    blocked = bool(Builder.set_val(ld, 'blocked', default = 0))
    return CrossMapResponseNormLayer(name, pow, size, scale, blocked)
  
#@util.lazyinit(distnet.init)
def add_layers(builder, net, model):
  net.append_layer(DataLayer('data0', net.image_shape))
  for layer in model:
    l = builder.make_layer(net, layer)
    if l is not None:
      net.append_layer(l)

def is_cudaconvnet_config(model):
  for layer in model:
    if 'filters' in layer or 'channels' in layer:
      return True
  return False
