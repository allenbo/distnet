from pycuda import driver, autoinit
import os
import cPickle as pickle
import reader
import numpy as np
from util import divup
from state import State, combination_conv, combination_fc, state0, disw_i
import request
from execute import RequestExecuter
from communicat import ConvFC, comm_cost

def device_name():
  return driver.Device(0).name().replace(' ', '')


def computation_cost(model, image_shape, comp_cost, req = None):
  conv_end = False
  comb = combination_conv
  input_shape = image_shape
  for layer in model:
    layer['input_shape'] = input_shape
    layer['overlapping'] = 0

    if layer['type'] == 'conv':
      channel, image_size, image_size, batch_size = input_shape

      padding = layer['padding']
      filter_size = layer['filterSize']
      stride = layer['stride']
      num_filter = layer['numFilter']
      layer['weight_size'] = filter_size * filter_size * num_filter * channel * 4
      layer['weight_shape'] = (channel, filter_size, filter_size, num_filter)

      output_size = 1 + divup(2 * padding + image_size - filter_size, stride)
      layer['output_shape'] = (num_filter, output_size, output_size, batch_size)

      
 
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
      layer['input_shape'] = (image_size, batch_size)
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
    
    input_shape = layer['output_shape']
    
    if conv_end == True:
      comb = combination_fc
    
    if comp_cost is not None:
      for s in comb:
        layer['comp_cost'][(s, n)] = comp_cost[layer['name']][(s,n)]
        layer['overlapping'] = comp_cost[layer['name']].get('overlapping', 0)

    else:
      if req == None:
        assert False, 'Have to specify a file'
      for s in comb:
        req.write_request(layer, s, n, conv_end)


def find_best(model, init_state, cfs):
  if len(model) == 0:
    return (0, [])
  layer = model[0]

  costs = []
  states = []
  
  input_size = layer.get('input_size', 0)
  weight_size = layer.get('weight_size', 0)
  overlapping = layer.get('overlapping', 0)

  communicat_cost = comm_cost[cfs]
  comb = combination_conv
  if cfs != ConvFC.conv:
    comb = combination_fc

  if cfs == ConvFC.conv_fc or cfs == ConvFC.fc:
    cfs = ConvFC.fc
  else:
    next_layer = model[1]
    if next_layer['type'] == 'fc':
      cfs = ConvFC.conv_fc
    else:
      cfs = ConvFC.conv

  if layer['type'] not in ['fc', 'conv', 'softmax']:
    cost, state_list = find_best(model[1:], init_state, cfs)
    communicat_cost = 0 if init_state != disw_i else overlapping * 2
    return (layer['comp_cost'][(init_state, n)] + communicat_cost * 1.0 / bandwidth + cost, [init_state] + state_list)

  for s in comb:
    cost, state_list = find_best(model[1:], s, cfs)
    cm_cost = communicat_cost[(init_state, s)](input_size, weight_size, overlapping, n) * 1.0 / bandwidth
    cp_cost = layer['comp_cost'][(s, n)]
    cost = cm_cost + cp_cost + cost
    costs.append(cost)
    states.append(state_list)
  
    index = np.array(costs).argmin()
  return (costs[index], [comb[index]] + states[index])
    
  
name = device_name()
n = 2

model_file = '../config/imagenet.cfg'
image_shape = (3, 224, 224, 128)
bandwidth = 1e9

model = reader.getmodel(model_file)
filename = '%s-%d.%s' % (name, n, os.path.basename(model_file))
if not os.path.exists(filename):
  req_filename = filename + '-req'
  with open(req_filename, 'w') as fout:
    req =  request.RequestProxy(fout)
    computation_cost(model, image_shape, None, req)
    req.finish()
  executer = RequestExecuter(req_filename, filename)
  executer.execute()

with open(filename) as f:
  comp_cost = pickle.load(f)
computation_cost(model, image_shape, comp_cost)
cost, states = find_best(model, state0, ConvFC.conv)
print 'cost', cost
for i, state in enumerate(states):
  print model[i]['name'], state
