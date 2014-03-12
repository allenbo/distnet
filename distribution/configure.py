from pycuda import driver, autoinit
import os
import pickle
import reader
import numpy as np
from util import divup

class State(object):
  dist   = 'dist'
  dist_b = 'dist-by-batch'
  dist_i = 'dist-by-image'
  dist_f = 'dist-by-first'
  shared = 'shared'


state0 = (0, 0)
sisw = (State.shared, State.shared)
#for conv layer
sidw = (State.shared, State.dist)
disw_i = (State.dist_i, State.shared)
#for fc layer
sidw_f = (State.shared, State.dist_f)
#for both
disw_b = (State.dist_b, State.shared)

combination_conv =(sisw, sidw, disw_b, disw_i)
combination_fc = (sisw, sidw_f, disw_b)


def device_name():
  return driver.Device(0).name().replace(' ', '')


def computation_cost(model, image_shape, comp_cost, fout = None):
  conv_end = False
  comb = combination_conv
  input_shape = image_shape
  for layer in model:
    layer['input_shape'] = input_shape

    if layer['type'] == 'conv':
      channel, image_size, image_size, batch_size = input_shape

      padding = layer['padding']
      filter_size = layer['filterSize']
      stride = layer['stride']
      num_filter = layer['numFilter']
      layer['weight_size'] = filter_size * filter_size, num_filter * channel * 4
      layer['weight_shape'] = (channel, filter_size, filter_size, num_filter)

      output_size = 1 + divup(2 * padding + image_size - filter_size, stride)
      layer['output_shape'] = (num_filter, output_size, output_size, batch_size)

      layer['overlapping'] = 0
      
 
    elif layer['type'] in ['rnorm', 'cmrnorm', 'neuron']:
      layer['output_shape'] = input_shape

    elif layer['type'] == 'pool':
      channel, image_size, image_size, batch_size = input_shape
      pool_size, stride, start = layer['poolSize'], layer['stride'], layer['start']
      output_size = 1 + divup(image_size - pool_size - start, stride)

      layer['output_shape'] = (channel, output_size, output_size, batch_size)

    elif layer['type'] == 'fc':
      image_size = np.prod(input_shape[:-1])
      batch_size = input_shape[-1]
      output_size = layer['outputSize']
      layer['weight_shape'] = (output_size, image_size)
      layer['weight_size'] = np.prod(layer['weight_shape']) * 4
      layer['output_shape'] = (output_size, batch_size)
      
      conv_end = True
    elif layer['type'] == 'softmax':
      layer['output_shape'] = layer['input_shape']
    
    else:
      assert False, 'Layer Type Error %s' % layer['type']
    
    layer['input_size'] = np.prod(layer['input_shape']) * 4
    layer['output_size'] = np.prod(layer['output_shape']) * 4
    layer['comp_cost'] = {}
    
    if conv_end == True:
      comb = combination_fc
    
    if comp_cost is not None:
      for s in comb:
        layer['comp_cost'][(s, n)] = comp_cost[layer['name']][(s,n)]

    else:
      if fout == None:
        assert False, 'Have to specify a file'
      for s in comb:
        print >> fout,'?req', layer['name'], layer['input_shape'], s



conv_conv_comm_cost = {}
conv_fc_comm_cost = {}
fc_fc_comm_cost = {}


class ConvFC:
  conv = 0
  conv_fc = 1
  fc = 2

def find_best(model, init_state, cfs):
  if len(model) == 0:
    return (0, [])
  layer = model[0]

  costs = []
  states = []
  
  input_size = layer.get('input_size', 0)
  weight_size = layer.get('weight_size', 0)
  overlapping = layer.get('overlapping', 0)

  if cfs == ConvFC.conv:
    comm_cost = conv_conv_comm_cost
  elif cfs == ConvFC.conv_fc:
    comm_cost = conv_fc_comm_cost
  else:
    comm_cost = fc_fc_comm_cost

  comb = combination_conv
  if cfs != ConvFC.conv:
    comb = combination_fc

  next_layer = model[1]
  if cfs == ConvFC.conv_fc or cfs == ConvFC.fc:
    cfs = ConvFC.fc
  else:
    if next_layer['type'] == 'fc':
      cfs = ConvFC.conv_fc
    else:
      cfs = ConvFC.conv

  for s in combinations:
    cost, state_list = find_best(model[1:], s, cfs)
    costs.append(comm_cost[(init_state, s)](input_size, weight_size, overlapping, n) +
        layer['comp_cost'][(s, n)])
    states.append(state_list)
 
  index = np.array(costs).argmin()
  return (costs[index], [combination_conv_conv[index]] + states[index])
    
  


name = device_name()
n = 4

model_file = '../config/imagenet.cfg'
image_shape = (3, 224, 224, 128)

model = reader.getmodel(model_file)
filename = '%s-%d' % (name, n)
if os.path.exists(filename):
  with open(filename) as f:
    dic = pickle.load(f)
  computation_cost(model, image_shape, comp_cost)
  cost, states = find_best(model, s0, ConvFC.conv)
else:
  with open(filename+'-req', 'w') as fout:
    computation_cost(model, image_shape, None, fout)
