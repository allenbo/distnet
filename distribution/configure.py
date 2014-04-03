import reader
from util import divup
from state import State, combination_conv, combination_fc, state0, disw_i, sisw, sidw, sidw_f
import request
from execute import RequestExecuter
from communicat import ConvFC, comm_cost
from communicat_worker import comm_cost as comm_cost_worker

from pycuda import driver, autoinit
import numpy as np

import os
import sys
import cPickle as pickle

def device_name():
  return driver.Device(0).name().replace(' ', '')


def computation_cost(model, image_shape, comp_cost, req = None):
  conv_end = False
  comb = combination_conv
  input_shape = image_shape
  #print '{:10}\t{:30}\t{:20}\t{:20}'.format('layer', 'distribution', 'cp_cost', 'num_worker')
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
        layer['comp_cost'][s] = comp_cost[layer['name']][s]
        layer['overlapping'] = comp_cost[layer['name']].get('overlapping', 0)
        layer['actual_data'] = comp_cost[layer['name']].get('actual_data', 0)
        #print '{:10}\t{:30}\t{:20}\t{:20}'.format(layer['name'], s, layer['comp_cost'][s][0], layer['comp_cost'][s][1])

    else:
      if req == None:
        assert False, 'Have to specify a file'
      for s in comb:
        req.write_request(layer, s, n, conv_end)


def find_best(model, init_state, cfs, prev_nw):
  if len(model) == 0:
    return (0, [])
  layer = model[0]

  input_size = layer.get('input_size', 0)
  weight_size = layer.get('weight_size', 0)
  overlapping = layer.get('overlapping', 0)
  actual_data = layer.get('actual_data', 0)


  comb = combination_conv
  if cfs != ConvFC.conv:
    comb = combination_fc

  if cfs == ConvFC.conv_fc or cfs == ConvFC.fc:
    ncfs = ConvFC.fc
  else:
    if len(model) != 1:
      next_layer = model[1]
      if next_layer['type'] == 'fc':
        ncfs = ConvFC.conv_fc
      else:
        ncfs = ConvFC.conv

  if layer['type'] not in ['fc', 'conv', 'softmax']:
    cur_nw = layer['comp_cost'][init_state][1]
    if layer['type'] == 'pool' and init_state == disw_i and cur_nw != prev_nw:
      communicat_cost = comm_cost_worker[cfs]
      cost, state_list = find_best(model[1:], init_state, ncfs, cur_nw)
      cm_cost = communicat_cost[(init_state, init_state)](input_size, weight_size, actual_data, cur_nw, prev_nw) * 1.0 / bandwidth + latency * 2
      return (layer['comp_cost'][init_state][0] + cm_cost + cost, [init_state] + state_list)
    else:
      communicat_cost = comm_cost[cfs]
      cost, state_list = find_best(model[1:], init_state, ncfs, cur_nw)
      cm_cost = 0 if init_state != disw_i or layer['type'] == 'neuron' else overlapping * 2.0 / bandwidth
      cm_latency = 0 if cm_cost == 0 else latency * 2
      return (layer['comp_cost'][init_state][0] + cm_cost + cm_latency + cost, [init_state] + state_list)

  costs = []
  states = []
  
  for s in comb:
    cur_nw = layer['comp_cost'][s][1]
    cost, state_list = find_best(model[1:], s, ncfs, cur_nw)
    if cur_nw == prev_nw or prev_nw == -1:
      communicat_cost = comm_cost[cfs]
      cm_cost = communicat_cost[(init_state, s)](input_size, weight_size, overlapping, cur_nw) * 1.0 / bandwidth
      cm_latency = 0 if cm_cost == 0 else latency * 2 * (1 if init_state == s == disw_i else (cur_nw - 1))
    else:
      communicat_cost = comm_cost_worker[cfs]
      cm_cost = communicat_cost[(init_state, s)](input_size, weight_size, actual_data, cur_nw, prev_nw) * 1.0 / bandwidth
      cm_latency = 0 if cm_cost == 0 else latency * 2 * (1 if init_state == s == disw_i else (cur_nw - 1))
    cp_cost = layer['comp_cost'][s][0]
    cost = cm_cost + cp_cost + cost + cm_latency
    costs.append(cost)
    states.append(state_list)
  
  index = np.array(costs).argmin()
  return (costs[index], [comb[index]] + states[index])

def print_details(model, states):
  assert len(model) == len(states)
  prev_state = state0
  total_comp_cost = 0
  total_comm_cost = 0
  cfs = ConvFC.conv
  
  prev_nw = -1
  print '\033[93m{:10}\t{:30}\t{:20}\t{:20}\t{:20}\033[0m'.format('layer', 'distribution', 'cp_cost', 'cm_cost', 'num_worker')
  for i in range(len(model)):
    layer = model[i]
    curr_state = states[i]

    input_size = layer.get('input_size', 0)
    weight_size = layer.get('weight_size', 0)
    overlapping = layer.get('overlapping', 0)
    actual_data = layer.get('actual_data', 0)
    cur_nw = layer['comp_cost'][curr_state][1]

    if cur_nw == prev_nw or prev_nw == -1:
      communicat_cost = comm_cost[cfs]
    else:
      communicat_cost = comm_cost_worker[cfs]

    if cfs == ConvFC.conv_fc or cfs == ConvFC.fc:
      cfs = ConvFC.fc
    else:
      if i + 1 != len(model):
        next_layer = model[i+1]
        if next_layer['type'] == 'fc':
          cfs = ConvFC.conv_fc
        else:
          cfs = ConvFC.conv

    if layer['type'] not in ['fc', 'conv', 'softmax']:
      if layer['type'] == 'pool' and curr_state == disw_i and cur_nw != prev_nw:
        cm_cost = communicat_cost[(prev_state, curr_state)](input_size, weight_size, actual_data, cur_nw, prev_nw) * 1.0 / bandwidth
      else:
        cm_cost = 0 if curr_state != disw_i or layer['type'] == 'neuron' else overlapping * 2.0 / bandwidth
      communicat_latency = 0 if cm_cost == 0 else latency * 2
      cm_cost += communicat_latency
    else:
      if cur_nw == prev_nw or prev_nw == -1:
        cm_cost = communicat_cost[(prev_state, curr_state)](input_size, weight_size, overlapping, cur_nw) * 1.0 / bandwidth
        cm_cost += 0 if cm_cost == 0 else latency * 2 * (1 if prev_state == curr_state == disw_i else (cur_nw - 1))
      else:
        cm_cost = communicat_cost[(prev_state, curr_state)](input_size, weight_size, actual_data, cur_nw, prev_nw) * 1.0 / bandwidth
        cm_cost += 0 if cm_cost == 0 else latency * 2 * (1 if prev_state == curr_state == disw_i else (cur_nw - 1))
    cp_cost = layer['comp_cost'][curr_state][0]

    print '{:10}\t{:30}\t{:20}\t{:20}\t{:20}'.format(layer['name'], curr_state, cp_cost, cm_cost, cur_nw)
    prev_state = curr_state
    prev_nw = cur_nw
    total_comp_cost += cp_cost
    total_comm_cost += cm_cost
  print 'total computation cost is', total_comp_cost
  print 'total communication cost is', total_comm_cost
  print 'total cost is', total_comp_cost + total_comm_cost

name = device_name()
latency = 0.001

model_file = '../config/imagenet_many_filter.cfg'
strategy_file = 'strategy'
image_shape = (3, 224, 224, 128)
bandwidth = 2.5e9
n = int(sys.argv[1])

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
cost, states = find_best(model, state0, ConvFC.conv, -1)
print 'Total cost is \033[44m%f\033[0m' % cost
#states = [disw_i] * (len(model) - 6) + [sidw_f] * 5 + [sisw]
#states = [sisw] * len(model)
states = [sidw] * (len(model) - 6) + [sidw_f] * 4 + [sisw] *2
print_details(model, states)

strategy = {}
for i, layer in enumerate(model):
  strategy[layer['name']] = states[i]

with open(strategy_file, 'w') as fout:
    pickle.dump(strategy, fout)
