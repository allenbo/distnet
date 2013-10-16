from fastnet import util, layer
from fastnet.layer import TRAIN, WeightedLayer, TEST
from fastnet.util import timer
from pycuda import cumath, gpuarray, driver
from pycuda.gpuarray import GPUArray
import numpy as np
import sys
import time

class FastNet(object):
  def __init__(self, image_shape):
    self.batch_size = -1
    self.layers = []
    self.image_shape = image_shape

    self.numCase = self.cost = self.correct = 0.0
    self.numConv = 0
  
  def __getitem__(self, name):
    for layer in self.layers:
      if layer.name == name:
        return layer

  def __iter__(self):
    return iter(self.layers)

  def append_layer(self, layer):
    if self.layers:
      layer.attach(self.layers[-1])

    self.layers.append(layer)
    util.log_info('Append: %s  [%s] : %s', layer.name, layer.type, layer.get_output_shape())
    return layer

  def drop_layer_from(self, name):
    found = False
    for i, layer in enumerate(self.layers):
      if layer.name == name:
        found = True
        break

    if not found:
      util.log('Layer: %s not found.', name)
      return []

    return_layers = self.layers[i:]
    self.layers = self.layers[0:i]
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

  def fprop(self, data, train=TRAIN):
    assert len(self.layers) > 0, 'No outputs: uninitialized network!'

    input = data
    for layer in self.layers:
      layer.fprop(input, layer.output, train)
      input = layer.output
    
    return self.layers[-1].output

  def bprop(self, label, train=TRAIN):
    grad = label
    for i in range(1, len(self.layers) + 1):
      curr = self.layers[-i]
      if curr.disable_bprop: return
      prev = self.layers[-(i + 1)]     
      curr.bprop(grad, prev.output, curr.output, prev.output_grad)
      grad = prev.output_grad

  def update(self):
    for layer in self.layers:
      layer.update()

  def adjust_learning_rate(self, factor=1.0):
    util.log_info('Adjusting learning rate: %s', factor)
    for layer in self.layers:
      if isinstance(layer, WeightedLayer):
        layer.weight.epsilon *= factor
        layer.bias.epsilon *= factor

    self.print_learning_rates()

  def set_learning_rate(self, eps_w, eps_b):
    for layer in self.layers:
      if isinstance(layer, WeightedLayer):
        layer.weight.epsilon = eps_w
        layer.bias.epsilon = eps_w
    self.print_learning_rates()

  def print_learning_rates(self):
    util.log('Learning rates:')
    for layer in self.layers:
      if isinstance(layer, WeightedLayer):
        util.log('%s: %s %s %s', layer.name, layer.__class__.__name__, 
                 layer.weight.epsilon, layer.bias.epsilon)
      else:
        util.log('%s: %s', layer.name, layer.__class__.__name__)

  def clear_weight_incr(self):
    for l in self.layers:
      if isinstance(l, WeightedLayer):
        l.clear_incr()

  def get_cost(self, label, prediction):
    cost_layer = self.layers[-1]
    assert not np.any(np.isnan(prediction.get()))
    cost_layer.logreg_cost(label, prediction)
    return cost_layer.cost.get().sum(), cost_layer.batchCorrect

  def get_cost_multiview(self, label, prediction, num_view):
    cost_layer = self.layers[-1]
    assert not np.any(np.isnan(prediction.get()))
    cost_layer.logreg_cost_multiview(label, prediction, num_view)
    return cost_layer.cost.get().sum(), cost_layer.batchCorrect

  def get_batch_information(self):
    cost = self.cost
    numCase = self.numCase
    correct = self.correct
    self.cost = self.numCase = self.correct = 0.0
    return cost / numCase , correct / numCase, int(numCase)
  
  def get_batch_information_multiview(self, num_view):
    cost = self.cost
    numCase = self.numCase / num_view
    correct = self.correct
    self.cost = self.numCase = self.correct = 0.0
    return cost / numCase, correct / numCase, int(numCase)


  def get_correct(self):
    return self.layers[-1].get_correct()

  def prepare_for_train(self, data, label):
    timer.start()

    # If data size doesn't match our expected batch_size, reshape outputs.
    if data.shape[1] != self.batch_size:
      self.batch_size = data.shape[1]
      for layer in self.layers:
        layer.change_batch_size(self.batch_size)
        layer.init_output()

    if not isinstance(data, GPUArray):
      data = gpuarray.to_gpu(data).astype(np.float32)

    if not isinstance(label, GPUArray):
      label = gpuarray.to_gpu(label).astype(np.float32)

    label = label.reshape((label.size, 1))
    self.numCase += data.shape[1]

    return data, label

  def train_batch(self, data, label, train=TRAIN):
    data, label = self.prepare_for_train(data, label)
    prediction = self.fprop(data, train)
    cost, correct = self.get_cost(label, prediction)
    self.cost += cost
    self.correct += correct

    if train == TRAIN:
      self.bprop(label)
      self.update()

    # make sure we have everything finished before returning!
    # also, synchronize properly releases the Python GIL,
    # allowing other threads to make progress.
    driver.Context.synchronize()

  def test_batch_multiview(self, data, label, num_view):
    data, label = self.prepare_for_train(data, label)
    prediction = self.fprop(data, TEST)

    cost, correct = self.get_cost_multiview(label, prediction, num_view)
    self.cost += cost
    self.correct += correct

  def get_dumped_layers(self):
    return [l.dump() for l in self.layers]

  def disable_bprop(self):
    for l in self.layers:
      l.disable_bprop()

  def enable_bprop(self):
    for l in self.layers:
      l.enable_bprop()

  def get_report(self):
    pass

  def get_image_shape(self):
    return self.layers[0].get_output_shape()

  def get_learning_rate(self):
    return self.learning_rate

  def get_layer_by_name(self, layer_name):
    for l in self.layers:
      if l.name == layer_name:
        return l

    raise KeyError, 'Missing layer: %s' % layer_name

  def get_output_by_name(self, layer_name):
    for idx, l in enumerate(self.layers):
      if l.name == layer_name:
        return l.output

    raise KeyError, 'Missing layer: %s' % layer_name

  def get_output_index_by_name(self, layer_name):
    for idx, l in enumerate(self.layers):
      if l.name == layer_name:
        return idx

    raise KeyError, 'Missing layer: %s' % layer_name

  def get_output_by_index(self, index):
    return self.layers[index].output

  def get_first_active_layer_name(self):
    for layer in self.layers:
      if layer.disable_bprop == False and isinstance(layer, WeightedLayer):
        return layer.name
    return ''

  def get_weight_by_name(self, name):
    for layer in self.layers:
      if layer.name == name:
        return layer.weight.wt.get() + layer.bias.wt.get().transpose()

  def get_summary(self):
    sum = []
    for l in self.layers:
      if isinstance(l, WeightedLayer):
        sum.append(l.get_summary())
    return sum
