from fastnet import util, layer
from fastnet.cuda_kernel import gpu_copy_to, transpose
from fastnet.layer import ConvLayer, NeuronLayer, MaxPoolLayer, \
  ResponseNormLayer, FCLayer, SoftmaxLayer, TRAIN, WeightedLayer, TEST
from fastnet.parser import is_cudaconvnet_config, add_layers, FastNetBuilder, \
  CudaconvNetBuilder
from fastnet.util import timer
from pycuda import cumath, gpuarray, driver
from pycuda.gpuarray import GPUArray
import numpy as np
import sys
import time



class FastNetIterator:
  def __init__(self, layers):
    util.log('create a iterator')
    self.layers = layers
    self.ind = 0
  def next(self):
    if self.ind == len(self.layers):
      raise StopIteration
    else:
      layer = self.layers[self.ind]
      self.ind += 1
      return layer

class FastNet(object):
  def __init__(self, learningRate, imgShape, init_model = None):
    self.learningRate = learningRate
    self.numColor, self.imgSize, _ , self.batchSize = imgShape
    self.imgShapes = [imgShape]
    self.inputShapes = [(self.numColor * (self.imgSize ** 2), self.batchSize)]
    self.layers = []
    self.outputs = []
    self.grads = []
    self.output = None
    self.save_layers = None
    self.save_output = []

    self.numCase = self.cost = self.correct = 0.0

    self.numConv = 0

    if init_model is None:
      util.log('initial model not provided, network doesn\'t have any layer')
      return

    if 'layers' in init_model:
      # Loading from a checkpoint
      add_layers(FastNetBuilder(), self, init_model['layers'])
    else:
      if is_cudaconvnet_config(init_model):
        # AlexK config file
        add_layers(CudaconvNetBuilder(), self, init_model)
      else:
        # FastNet config file
        add_layers(FastNetBuilder(), self, init_model)
      self.adjust_learning_rate(self.learningRate)

    util.log('Learning rates:')
    for l in self.layers:
      if isinstance(l, WeightedLayer):
        util.log('%s: %s %s', l.name, getattr(l, 'epsW', 0), getattr(l, 'epsB', 0))

  def save_layerouput(self, layers):
    self.save_layers = layers

  def __getitem__(self, name):
    for layer in self.layers:
      if layer.name == name:
        return layer

  def __iter__(self):
    util.log('get iterator')
    return FastNetIterator(self.layers)

  def append_layer(self, layer):
    if self.layers:
      layer.attach(self.layers[-1])

    self.layers.append(layer)

    outputShape = layer.get_output_shape()
    row = outputShape[0] * outputShape[1] * outputShape[2]
    col = outputShape[3]
    self.inputShapes.append((row, col))
    self.imgShapes.append(outputShape)

    self.outputs.append(gpuarray.zeros((row, col), dtype=np.float32))
    self.grads.append(gpuarray.zeros(self.inputShapes[-2], dtype=np.float32))
    print >> sys.stderr,  '%s  [%s] : %s' % (layer.name, layer.type, outputShape)

  def drop_layer_from(self, name):
    for i, layer in enumerate(self.layers):
      if layer.name == name:
        break
    return_layers = self.layers[i:]
    self.layers = self.layers[0:i]
    del self.inputShapes[i+1:]
    del self.imgShapes[i:]
    del self.outputs[i:]
    del self.grads[i:]
    print 'delete layer from', name
    print 'the last layer would be', self.layers[-1].name
    return return_layers

  @staticmethod
  def split_conv_to_stack(conv_params):
    stack = {}
    s = []
    for ld in conv_params:
      if ld['type'] in ['fc', 'softmax']:
        break
      elif ld['type'] == 'conv':
        if s != []:
          stack[s[0]['name']] = s
        s = [ld]
      else:
        s.append(ld)
    stack[s[0]['name']] = s
    return stack

  @staticmethod
  def split_fc_to_stack(fc_params):
    stack = {}
    s = []
    for ld in fc_params:
      if ld['type'] == 'softmax':
        break
      elif ld['type'] == 'fc':
        if s != []:
          stack[s[0]['name']] = s
        s = [ld]
      else:
        s.append(ld)
    stack[s[0]['name']] = s
    return stack

  def fprop(self, data, probs, train=TRAIN):
    assert len(self.outputs) > 0, 'No outputs: uninitialized network!'

    input = data
    for i in range(len(self.layers)):
      l = self.layers[i]
      #if isinstance(l, layer.DataLayer):
      #  self.outputs[i] = input
      #else:
      l.fprop(input, self.outputs[i], train)

      input = self.outputs[i]

    # probs.shape = self.outputs[-1].shape
    gpu_copy_to(self.outputs[-1], probs)

  def bprop(self, data, label, prob, train=TRAIN):
    grad = label
    for i in range(1, len(self.layers) + 1):
      l = self.layers[-i]
      if l.disableBprop:
        return
      if i == len(self.layers):
        input = data
      else:
        input = self.outputs[-(i + 1)]
      output = self.outputs[-i]
      outGrad = self.grads[-i]
      l.bprop(grad, input, output, outGrad)
      grad = outGrad

  def update(self):
    for l in self.layers:
      if l.disableBprop or not isinstance(l, WeightedLayer):
        continue
      l.update()

  def adjust_learning_rate(self, factor=1.0):
    for layer in self.layers:
      if isinstance(layer, WeightedLayer):
        layer.scaleLearningRate(factor)


  def clear_weight_incr(self):
    for l in self.layers:
      if isinstance(l, WeightedLayer):
        l.clear_incr()

  def get_cost(self, label, output):
    outputLayer = self.layers[-1]
    outputLayer.logreg_cost(label, output)
    return outputLayer.cost.get().sum(), outputLayer.batchCorrect

  def get_batch_information(self):
    cost = self.cost
    numCase = self.numCase
    correct = self.correct
    self.cost = self.numCase = self.correct = 0.0
    return cost / numCase , correct / numCase, int(numCase)

  def get_correct(self):
    outputLayer = self.layers[-1]
    return outputLayer.get_correct()

  def prepare_for_train(self, data, label):
    timer.start()
    input = data

    # If data size doesn't match our expected batchsize, reshape outputs.
    if input.shape[1] != self.batchSize:
      self.batchSize = input.shape[1]
      for l in self.layers:
        l.change_batch_size(self.batchSize)
      self.inputShapes = None
      self.imgShapes = None
      self.outputs = []
      self.grads = []

      self.imgShapes = [(self.numColor, self.imgSize, self.imgSize, self.batchSize)]
      self.inputShapes = [(self.numColor * (self.imgSize ** 2), self.batchSize)]
      for layer in self.layers:
        # layer.update_shape(...)
        outputShape = layer.get_output_shape()
        row = outputShape[0] * outputShape[1] * outputShape[2]
        col = outputShape[3]
        self.inputShapes.append((row, col))
        self.imgShapes.append(outputShape)

        self.outputs.append(gpuarray.zeros((row, col), dtype=np.float32))
        self.grads.append(gpuarray.zeros(self.inputShapes[-2], dtype=np.float32))

    if not isinstance(data, GPUArray):
      data = gpuarray.to_gpu(data).astype(np.float32)

    if not isinstance(label, GPUArray):
      label = gpuarray.to_gpu(label).astype(np.float32)

    label = label.reshape((label.size, 1))
    self.numCase += input.shape[1]
    outputShape = self.inputShapes[-1]

    if self.output is None or self.output.shape != outputShape:
      self.output = gpuarray.zeros(outputShape, dtype=np.float32)

    return data, label

  def train_batch(self, data, label, train=TRAIN):
    data, label = self.prepare_for_train(data, label)
    self.fprop(data, self.output, train)
    cost, correct = self.get_cost(label, self.output)
    self.cost += cost
    self.correct += correct

    if self.save_layers is not None:
      it = [(i, self.layers[i].name) for i in range(len(self.layers)) if self.layers[i].name in self.save_layers]
      outputs = [transpose(o).get() for o in self.outputs]
      label = label.get()
      self.save_output.extend([(label[i, 0], dict([(name, outputs[j][i,:]) for j, name in it])) for i in range(self.batchSize)])

    if train == TRAIN:
      self.bprop(data, label, self.output)
      self.update()

    # make sure we have everything finished before returning!
    # also, synchronize properly releases the Python GIL,
    # allowing other threads to make progress.
    driver.Context.synchronize()

  def get_dumped_layers(self):
    layers = []
    for l in self.layers:
      layers.append(l.dump())

    return layers

  def get_save_output(self):
    if self.save_layers is None:
      assert False, 'Not specify any save layer name'
    else:
      tmp = self.save_output
      self.save_output = []
      return tmp

  def disable_bprop(self):
    for l in self.layers:
      l.disableBprop()

  def enable_bprop(self):
    for l in self.layers:
      l.enableBprop()

  def get_report(self):
    pass

  def get_image_shape(self):
    return self.imgShaps[0]

  def get_learning_rate(self):
    return self.learningRate

  def get_layer_by_name(self, layer_name):
    for l in self.layers:
      if l.name == layer_name:
        return l

    raise KeyError, 'Missing layer: %s' % layer_name

  def get_output_by_name(self, layer_name):
    for idx, l in enumerate(self.layers):
      if l.name == layer_name:
        return self.outputs[idx]

    raise KeyError, 'Missing layer: %s' % layer_name

  def get_output_index__by_name(self, layer_name):
    for idx, l in enumerate(self.layers):
      if l.name == layer_name:
        return idx

    raise KeyError, 'Missing layer: %s' % layer_name

  def get_output_by_index(self, index):
    return self.outputs[index]

  def get_summary(self):
    sum = []
    for l in self.layers:
      if isinstance(l, WeightedLayer):
        sum.append(l.get_summary())
    return sum


class AdaptiveFastNet(FastNet):
  def __init__(self, learningRate, imgShape, train, test, init_model):
    FastNet.__init__(self, learningRate, imgShape, init_model)
    self.train_data, self.train_label = train
    self.test_data, self.test_label = test
    self.adjust_info = [(self.learningRate, 0, 0)]

  def adjust_learning_rate(self, factor):
    factors = factor
    train_data = self.train_data
    test_data = self.test_data
    train_label = self.train_label
    test_label = self.test_label

    weights = []
    biases = []
    epsW = []
    epsB = []

    print 'store the weight, bias and learning rate'
    for layer in self.layers:
      if isinstance(layer, WeightedLayer):
        weight = gpuarray.empty_like(layer.weight)
        gpu_copy_to(layer.weight, weight)
        weights.append(weight)
        epsW.append(layer.epsW)

        bias = gpuarray.empty_like(layer.bias)
        gpu_copy_to(layer.bias, bias)
        biases.append(bias)
        epsB.append(layer.epsB)

    print 'find the best learning rate'
    print 'the factor list is ', factors

    self.prepare_for_train(train_data, train_label)
    self.fprop(self.data, self.output)
    self.bprop(self.data, self.label, self.output)

    self.get_batch_information()
    self.update()

    self.train_batch(test_data, test_label, TEST)
    cost, correct, numCase = self.get_batch_information()
    best = (correct , 1.0)
    origin = (correct, 1.0)
    print 'The normal update produce the correct', correct, 'number of case is', numCase

    for factor in factors:
      print 'Try the factor', factor
      i = 0
      for layer in self.layers:
        if isinstance(layer, WeightedLayer):
          gpu_copy_to(weights[i], layer.weight)
          gpu_copy_to(biases[i], layer.bias)
          layer.epsW = epsW[i] * factor
          layer.epsB = epsB[i] * factor
          i += 1

      self.update()
      '''
      for layer in self.layers:
        if isinstance(layer, WeightedLayer):
          print 'epsW', layer.epsW, 'epsB', layer.epsB
          printMatrix(layer.weight, layer.name + 'weight')
          printMatrix(layer.bias, layer.name + 'bias')
      '''
      self.train_batch(test_data, test_label, TEST)
      cost, correct, numCase = self.get_batch_information()
      print 'Applying factor', factor, ', The correct is', correct, 'number of case is', numCase
      if correct > best[0]:
        best = (correct, factor)

    self.adjust_info.append((best[1], best[0], origin[0]))
    if best[0] / origin[0] < 1.025:
      best = origin
    factor = best[1]
    i = 0
    for layer in self.layers:
      if isinstance(layer, WeightedLayer):
        gpu_copy_to(weights[i], layer.weight)
        gpu_copy_to(biases[i], layer.bias)
        layer.epsW = epsW[i] * factor
        layer.epsB = epsB[i] * factor
        print 'Layer', layer.name
        print 'epsW is', layer.epsW, 'epsB is', layer.epsB
        i += 1
    # self.update()

  def get_report(self):
    return self.adjust_info

