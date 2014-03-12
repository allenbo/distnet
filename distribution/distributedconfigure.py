import numpy as np
import sys


class SettingReader(object):
  @staticmethod
 ; def read(file):
    rst = {}
    with open(file) as f:
      for line in f:
        line = line.strip()
        if line.startswith('#'):
          continue
        else:
          key = line[0:line.find('=')]
          value = line[line.find('=')+1: len(line)]

          if value.isdigit():
            value = int(value)
          elif isfloat(value):
            value = float(value)
          else:
            value = eval(value)

          rst[key] = value
    return rst


def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def isinteger(value):
  try:
    int(value)
    return True
  except ValueError:
    return False


def getmodel(modelfile):
  rst = []
  with open(modelfile) as f:
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
        elif isfloat(value):
          value = float(value)

        rst[-1][key] = value
  return rst


def divup(x, base):
  if x / base * base == x:
    return int(x / base)
  else:
    return int(x / base + 1)



def getlocalcost(model, image_shape):
  input_shape = image_shape
  for layer in model:
    #print layer['name']
    if layer['type'] == 'conv':
      channel, image_size, image_size, batch_size = input_shape
      layer['input_shape'] = input_shape
      padding = layer['padding']
      filter_size = layer['filterSize']
      stride = layer['stride']
      num_filter = layer['numFilter']
      layer['weight_size'] = filter_size * filter_size * num_filter * channel * 4
      output_size = 1 + divup(2 * padding + image_size - filter_size, stride)
      layer['output_shape'] = (num_filter, output_size, output_size, batch_size)
      module_size = output_size * output_size * num_filter
      input_pixel = image_size * image_size * channel
      # fprop computation
      layer['computation_cost'] = filter_size * filter_size * module_size * batch_size * channel
      # bprop computation
      layer['computation_cost'] *= 2.8
      #print 'computation amount', layer['computation_cost']

      layer['computation_cost']  /= GPUcomputability * 0.1
      #print 'computation cost', layer['computation_cost']

    elif  layer['type'] == 'pool':
      channel, image_size, image_size, batch_size =  input_shape
      layer['input_shape'] = input_shape
      pool_size = layer['poolSize']
      stride = layer['stride']
      start = layer['start']
      output_size =  divup(image_size - pool_size- start, stride) + 1
      layer['output_shape'] = (channel, output_size, output_size, batch_size)

      # fprop computation
      layer['computation_cost'] = pool_size * pool_size * output_size * output_size * channel * batch_size
      # bprop computation
      layer['computation_cost'] *= 2
      #print 'computation amount', layer['computation_cost']

      layer['computation_cost'] /= GPUcomputability * 0.1
      #print 'computation cost', layer['computation_cost']


    elif layer['type'] == 'rnorm':
      layer['input_shape'] = input_shape
      channel, image_size, image_size, batch_size =  input_shape
      size = layer['size']
      layer['output_shape'] = (channel, image_size, image_size, batch_size)

      layer['computation_cost'] = size * size * image_size * image_size * channel * batch_size
      layer['computation_cost'] *= 2
      #print 'computation amount', layer['computation_cost']
      layer['computation_cost'] /= GPUcomputability* 0.1
      #print 'computation cost', layer['computation_cost']

    elif layer['type'] == 'cmrnorm':
      layer['input_shape'] = input_shape
      channel, image_size, image_size, batch_size =  input_shape
      size = layer['size']
      layer['output_shape'] = (channel, image_size, image_size, batch_size)

      layer['computation_cost'] = size * image_size * image_size * channel * batch_size
      layer['computation_cost'] *= 2
      #print 'computation amount', layer['computation_cost']
      layer['computation_cost'] /= GPUcomputability * 0.1
      #print 'computation cost', layer['computation_cost']

    elif layer['type'] == 'fc':
      layer['input_shape'] = input_shape
      image_size = np.prod(input_shape[:-1])
      batch_size = input_shape[-1]
      output_size = layer['outputSize']
      layer['weight_size'] = image_size * output_size * 4
      layer['output_shape'] = (output_size, batch_size)
      layer['computation_cost'] = output_size * batch_size * image_size * 3

      #print 'computation amount', layer['computation_cost']
      layer['computation_cost'] /= GPUcomputability * 1.0
      #print 'computation cost', layer['computation_cost']

    elif layer['type'] == 'softmax':
      layer['input_shape'] = input_shape
      input_size, batch_size = input_shape
      layer['output_shape'] = (input_size, batch_size)
      layer['computation_cost'] = 0.0

    elif layer['type'] == 'neuron':
      layer['input_shape'] = input_shape
      layer['output_shape'] = tuple(input_shape[:])
      layer['computation_cost'] = 0.0

    input_shape = layer['output_shape']


class State(object):
  dist   = 'dist'
  dist-b = 'dist-by-batch'
  dist-i = 'dist-by-image'
  dist-f = 'dist-by-first'
  shared = 'shared'


state0 = (0, 0)
sisw = (State.shared, State.shared)
#for conv layer
sidw = (State.shared, State.dist)
disw-i = (State.dist-i, State.shared)
#for fc layer
sidw-f = (State.shared, State.dist-f)
#for both
disw-b = (State.dist-b, State.shared)

combinations = (disw, sidw, sisw)


computation_dict = {
    disw : lambda comp_cost, n: comp_cost / n,
    sidw : lambda comp_cost, n: comp_cost / n,
    sisw : lambda comp_cost, n: comp_cost
}

communication_dict = {
    (state0, disw) : lambda input_size, weight_size, n: 2.0*(n-1)*weight_size / bandwidth, #input_size / disk_bandwidth + n * (n -1) / 2 * weight_size,
    (state0, sidw) : lambda input_size, weight_size, n: 0, #n * input_size / disk_bandwidth,
    (state0, sisw) : lambda input_size, weight_size, n: 0, #n * input_size / disk_bandwidth,
    (disw, disw) : lambda input_size, weight_size, n: 2.0*(n-1)*weight_size / bandwidth,
    (disw, sidw) : lambda input_size, weight_size, n: 2.0*(n*n-1)/n * input_size / bandwidth,
    (disw, sisw) : lambda input_size, weight_size, n: 2.0*(n-1)/n  * input_size / bandwidth,
    (sidw, sidw) : lambda input_size, weight_size, n: 2.0*(n*n-1)/n * input_size / bandwidth,
    (sidw, disw) : lambda input_size, weight_size, n: (2.0*(n - 1)/n* input_size + 2.0*(n-1)/n* weight_size) / bandwidth,
    (sidw, sisw) : lambda input_size, weight_size, n: 2.0*(n - 1) / n * input_size / bandwidth,
    (sisw, sisw) : lambda input_size, weight_size, n: 0 ,
    (sisw, sidw) : lambda input_size, weight_size, n: 2.0*(n -1) * input_size / bandwidth,
    (sisw, disw) : lambda input_size, weight_size, n: (2.0*(n-1)/n * input_size + 2.0*(n -1) * weight_size) / bandwidth
}


def find_best(model, init_state):
  if len(model) == 0:
    return (0, [])
  layer = model[0]
  computation_cost = layer['computation_cost']
  if layer['type'] not in ['fc', 'conv']:
    cost, state_list = find_best(model[1:], init_state)
    return (computation_dict[init_state](computation_cost, num_worker) + cost , [init_state] + state_list)

  input_size = np.prod(layer['input_shape'])
  weight_size = layer['weight_size']
  costs = []
  states = []
  for s in combinations:
    cost, state_list = find_best(model[1:], s)
    costs.append(communication_dict[(init_state, s)](input_size, weight_size, num_worker) + computation_dict[s](computation_cost, num_worker) + cost)
    states.append(state_list)

  index = np.array(costs).argmin()
  return (costs[index], [combinations[index]] + states[index])



def print_details(model, states):
  assert len(model) == len(states)
  prev_state = state0
  total_comp_cost = 0
  total_comm_cost = 0
  for i in range(len(model)):
    layer = model[i]
    curr_state = states[i]

    input_size = np.prod(layer['input_shape'])
    if layer['type'] in ['conv', 'fc']:
      weight_size =  layer['weight_size']
      comm_cost = communication_dict[(prev_state, curr_state)](input_size, weight_size, num_worker)
    else:
      comm_cost = 0
    prev_state = curr_state
    comp_cost = computation_dict[curr_state](layer['computation_cost'], num_worker)
    print layer['name'], curr_state, comm_cost
    total_comp_cost += comp_cost
    total_comm_cost += comm_cost
  print 'total computation cost is', total_comp_cost
  print 'total communication cost is', total_comm_cost
  print 'total cost is', total_comp_cost + total_comm_cost



if len(sys.argv) != 3:
  print 'usage: python %s <setting_file> <model_config>' % __file__
  sys.exit(1)


modelconfigure_filename = sys.argv[2]
dic = SettingReader.read(sys.argv[1])
num_worker = dic.get('num_worker', 1)
GPUcomputability =  dic.get('GPUcomputability', 1.0*1024*1024*1024)
latency = dic.get('latency', 0.005)
bandwidth = dic.get('bandwidth', 128*1024*1024)
disk_bandwidth = dic.get('disk_bandwidth', 60*1024*1024)

image_shape = (3, 224, 224, 128)
model = getmodel(modelconfigure_filename)
getlocalcost(model, image_shape)
states = [
#    ('shared', 'shared'), # conv1
#    ('shared', 'shared'), # neuron1
#    ('shared', 'shared'), # pool1
#    ('shared', 'shared'), # rnorm1
#    ('shared', 'shared'), # conv2
#    ('shared', 'shared'), # neuron2
#    ('shared', 'shared'), # pool2
#    ('shared', 'shared'), # rnorm2
#    ('shared', 'shared'), # conv3
#    ('shared', 'shared'), # neuron3
#    ('shared', 'shared'), # conv4
#    ('shared', 'shared'), # neuron4
#    ('shared', 'shared'), # conv5
#    ('shared', 'shared'), # neuron5
#    ('shared', 'shared'), # pool5
#    ('shared', 'shared'), # fc6
#    ('shared', 'shared'), # neuron6
#    ('shared', 'shared'), # fc7
#    ('shared', 'shared'), # neuron7
#    ('shared', 'shared'), # fc8
#    ('shared', 'shared'), # softmax
('shared', 'dist'), 
('shared', 'dist'), 
('shared', 'dist'), 
('shared', 'dist'), 
('dist', 'shared'), 
('dist', 'shared'), 
('dist', 'shared'),
('dist', 'shared'),
('shared', 'shared'), 
('shared', 'shared'), 
('shared', 'shared'), 
('shared', 'shared'), 
('shared', 'shared'), 
('shared', 'shared'), 
('shared', 'shared'), 
('shared', 'shared'), 
('shared', 'shared'), 
('shared', 'shared'), 
('shared', 'shared'), 
('shared', 'shared'), 
('shared', 'shared'), 
]
cost, states = find_best(model, state0)
print cost
print_details(model, states)
