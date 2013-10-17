from fastnet import net, util
from fastnet.layer import ConvLayer, MaxPoolLayer, AvgPoolLayer, \
  CrossMapResponseNormLayer, SoftmaxLayer, NeuronLayer, ResponseNormLayer, FCLayer, \
  DataLayer
import fastnet
import numpy as np

def parse_config_file(parsing_file):
  rst = []
  with open(parsing_file) as f:
    for line in f:
      line = line.strip()
      if line.startswith('#'):
        continue
      elif line.startswith('['):
        name = line[1:line.find(']')]
        rst.append({'name':name})
      elif len(line) == 0:
        continue
      else:
        key = line[0:line.find('=')]
        value = line[line.find('=')+1: len(line)]

        if value.isdigit():
          value = int(value)
        elif util.isfloat(value):
          value = float(value)

        rst[-1][key] = value
  return rst

def load_model(net, model):
  if 'layers' in model:
    util.log('Loading from checkpoint...')
    # Loading from a checkpoint
    add_layers(FastNetBuilder(), net, model['layers'])
  else:
    #net.append_layer(DataLayer('data0', net.image_shape)) 
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

class OptHelper():
  '''
  Builds up a list of valid option names, which are checked on deletion.
  '''
  def __init__(self, load_dict):
    self.load_dict = {}
    self.valid_dic = {}
  
  def get(self, ld, name, default = None):
    if default is not None:
      val = ld.get(name, default)
    else:
      val = ld.get(name)
      
    self.valid_dic[name] = 1
    return val
  
  def __del__(self):
    for k in self.load_dict.keys():
      assert k in self.valid_dic, 'Unknown key: %s' % k
  

class Builder(object):
  def make_layer(self, net, ld):
    if ld['type'] == 'conv': return self.conv_layer(ld)
    elif ld['type'] == 'pool': return self.pool_layer(ld)
    elif ld['type'] == 'neuron': return self.neuron_layer(ld)
    elif ld['type'] == 'fc': return self.fc_layer(ld)
    elif ld['type'] == 'softmax': return self.softmax_layer(ld)
    elif ld['type'] == 'rnorm': return self.rnorm_layer(ld)
    elif ld['type'] == 'cmrnorm': return self.crm_layer(ld)
    elif ld['type'] == 'data':
      return DataLayer(ld['name'], ld['image_shape'])
    else:
      raise Exception, 'Unknown layer %s' % ld['type']


class FastNetBuilder(Builder):
  def conv_layer(self, ld):
    opts = OptHelper(ld)
    
    numFilter = opts.get(ld, 'numFilter')
    filterSize = opts.get(ld, 'filterSize')
    padding = opts.get(ld, 'padding')
    stride = opts.get(ld, 'stride')
    initW = opts.get(ld, 'initW', 0.01)
    initB = opts.get(ld, 'initB', 0.00)
    epsW = opts.get(ld, 'epsW', 0.001)
    epsB = opts.get(ld, 'epsB', 0.002)
    if epsB == 0:
      epsB = 0.002
    momW = opts.get(ld, 'momW', 0.0)
    momB = opts.get(ld, 'momB', 0.0)
    sharedBiases = opts.get(ld, 'sharedBiases', default = 1)
    partialSum = opts.get(ld, 'partialSum', default = 0)
    wc = opts.get(ld, 'wc', 0.0)
    bias = opts.get(ld, 'bias')
    weight = opts.get(ld, 'weight')
    weightIncr = opts.get(ld, 'weightIncr')
    biasIncr = opts.get(ld, 'biasIncr')
    name = opts.get(ld, 'name')
    disable_bprop = opts.get(ld, 'disable_bprop', default = False)
    cv = ConvLayer(name, numFilter, (filterSize, filterSize), padding, stride, initW, initB,
        partialSum,sharedBiases, epsW, epsB, momW, momB, wc, bias, weight,
        weightIncr = weightIncr, biasIncr = biasIncr, disable_bprop = disable_bprop)
    return cv

  def pool_layer(self, ld):
    opts = OptHelper(ld)
    stride = opts.get(ld, 'stride')
    start = opts.get(ld, 'start')
    poolSize = opts.get(ld, 'poolSize')
    name = opts.get(ld, 'name')
    pool = opts.get(ld, 'pool', default = 'max')
    disable_bprop = opts.get(ld, 'disable_bprop', default = False)
    if pool == 'max':
      return MaxPoolLayer(name, poolSize, stride, start, disable_bprop = disable_bprop)
    elif pool == 'avg':
      return AvgPoolLayer(name, poolSize, stride, start, disable_bprop = disable_bprop)

  def crm_layer(self, ld):
    opts = OptHelper(ld)
    name = opts.get(ld, 'name')
    pow = opts.get(ld, 'pow')
    size = opts.get(ld, 'size')
    scale = opts.get(ld, 'scale')
    blocked = bool(opts.get(ld, 'blocked', default = 0))
    disable_bprop = opts.get(ld, 'disable_bprop', default = False)
    return CrossMapResponseNormLayer(name, pow, size, scale, blocked, disable_bprop =
        disable_bprop)

  def softmax_layer(self, ld):
    opts = OptHelper(ld)
    name = opts.get(ld, 'name')
    disable_bprop = opts.get(ld, 'disable_bprop', default = False)
    return SoftmaxLayer(name, disable_bprop = disable_bprop)

  def neuron_layer(self, ld):
    opts = OptHelper(ld)
    name = opts.get(ld, 'name')
    disable_bprop = opts.get(ld, 'disable_bprop', default = False)
    if ld['neuron'] == 'relu':
      e = opts.get(ld, 'e')
      return NeuronLayer(name, type='relu', e=e, disable_bprop = disable_bprop)

    if ld['neuron'] == 'tanh':
      a = opts.get(ld, 'a')
      b = opts.get(ld, 'b')
      return NeuronLayer(name, type='tanh', a=a, b=b, disable_bprop = disable_bprop)

    assert False, 'No implementation for the neuron type' + ld['neuron']['type']

  def rnorm_layer(self, ld):
    opts = OptHelper(ld)
    name = opts.get(ld, 'name')
    pow = opts.get(ld,'pow')
    size = opts.get(ld, 'size')
    scale = opts.get(ld, 'scale')
    disable_bprop = opts.get(ld, 'disable_bprop', default = False)
    return ResponseNormLayer(name, pow, size, scale, disable_bprop = disable_bprop)


  def fc_layer(self, ld):
    opts = OptHelper(ld)
    epsB = opts.get(ld, 'epsB', 0.002)
    if epsB == 0:
      epsB = 0.002
    epsW = opts.get(ld ,'epsW', 0.001)
    initB = opts.get(ld, 'initB', 0.00)
    initW = opts.get(ld, 'initW', 0.01)
    momB = opts.get(ld, 'momB', 0.0)
    momW = opts.get(ld, 'momW', 0.0)
    wc = opts.get(ld, 'wc', 0.0)
    dropRate = opts.get(ld, 'dropRate', 0.0)

    n_out = opts.get(ld , 'outputSize')
    bias = opts.get(ld, 'bias')
    weight = opts.get(ld, 'weight')
    #if isinstance(weight, list):
    #  weight = np.concatenate(weight)

    weightIncr = opts.get(ld, 'weightIncr')
    biasIncr = opts.get(ld, 'biasIncr')
    name = opts.get(ld, 'name')
    disable_bprop = opts.get(ld, 'disable_bprop', default = False)
    return FCLayer(name, n_out, epsW, epsB, initW, initB, momW, momB, wc, dropRate,
        weight, bias, weightIncr = weightIncr, biasIncr = biasIncr, disable_bprop = disable_bprop)




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
    opts = OptHelper(ld)
    name = opts.get(ld, 'name')
    pow = opts.get(ld,'pow')
    size = opts.get(ld, 'size')
    scale = opts.get(ld, 'scale')
    scale = scale * size ** 2
    return ResponseNormLayer(name, pow, size, scale)

  def crm_layer(self, ld):
    opts = OptHelper(ld)
    name = opts.get(ld, 'name')
    pow = opts.get(ld, 'pow')
    size = opts.get(ld, 'size')
    scale = opts.get(ld, 'scale')
    scale = scale * size
    blocked = bool(opts.get(ld, 'blocked', default = 0))
    return CrossMapResponseNormLayer(name, pow, size, scale, blocked)
  
@util.lazyinit(fastnet.init) 
def add_layers(builder, net, model):
  # have to handle the data layer specially, as it is not saved properly 
  # in some checkpoints
  data_layer = model[0]
  #if data_layer['type'] == 'data' and 'image_shape' in data_layer:
  #  net.append_layer(builder.make_layer(net, data_layer))
  #else:
  #  net.append_layer(DataLayer('data0', net.image_shape))
  if data_layer['type'] != 'data':
    net.append_layer(DataLayer('data0', net.image_shape))
  elif 'image_shape' not in data_layer:
    data_layer['image_shape'] = net.image_shape

  for layer in model:
    l = builder.make_layer(net, layer)
    if l is not None:
      net.append_layer(l)

def is_cudaconvnet_config(model):
  for layer in model:
    if 'filters' in layer or 'channels' in layer:
      return True
  return False
