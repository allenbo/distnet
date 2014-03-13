from distnet.layer import ConvLayer
from util import divup, make_square_shape, issquare
from state import sisw, sidw, disw_i, sidw_f, disw_b
import copy
import StringIO
import json
from fake_varray import VArray


class Request(object):
  def __init__(self, id, name, type, num_worker, state, list):
    self.id = id
    self.name = name
    self.type = type
    self.list = list
    self.num_worker = num_worker
    self.state = state

  def write_request(self, fout):
    string_out = StringIO.StringIO()
    decr_dic = {'id': self.id, 'op':self.name, 'type':self.type, 'workers':self.num_worker,
        'state':self.state, 'tests':len(self.list)}
    item = {'decr': decr_dic, 'param': self.list}
    json.dump(item, string_out, sort_keys = True, indent = 2, separators = (',', ': '))
    #json.dump(item, string_out)
    print >> fout , string_out.getvalue(), ','

class RequestWriter(object):
  def __init__(self, layer, state, num_worker):
    self.state = state
    self.num_worker = num_worker
    self.name = layer['type']

    self.input_shape = layer['input_shape']
    self.output_shape = layer['output_shape']
    self.batch_size = self.input_shape[-1]
    self.dict = {
        'input_shape':self.input_shape,
        'output_shape':self.output_shape,
        }
    self.list = []

  def define_request(self):
    assert False, 'Implementation missing in RequestWriter'

  def define_disw_b(self):
    num_image = divup(self.batch_size, self.num_worker)
    last_num_image = self.batch_size - num_image * (self.num_worker - 1)
    dict = copy.deepcopy(self.dict)
    dict['input_shape'] = self.input_shape[:-1] + (num_image, )
    dict['output_shape'] = self.output_shape[:-1] + (num_image, )
    self.list.append(dict)

    if num_image != last_num_image:
      self.type = 'max'
      dict2 = copy.deepcopy(dict)
      dict2['input_shape'] = self.input_shape[:-1] + (last_num_image, )
      dict2['output_shape'] = self.output_shape[:-1] + (last_num_image, )
      self.list.append(dict2)


  def write_request(self, id, fout):
    self.define_request()
    Request(id, self.name, self.type, self.num_worker, self.state, self.list).write_request(fout)


class ConvRequestWriter(RequestWriter):
  ''' This is not just for conv layer, it should work for conv stack'''
  def __init__(self, layer, state, num_worker):
    RequestWriter.__init__(self, layer, state, num_worker)
    if self.name == 'conv':
      self.padding, self.filter_size, self.stride = layer['padding'], layer['filterSize'], layer['stride']
      self.num_filter = layer['numFilter']
      self.weight_shape = layer['weight_shape']
      self.dict['filter_shape'] = self.weight_shape
      self.dict['stride'] = self.stride
      self.dict['padding'] = self.padding
    elif self.name == 'pool':
      self.filter_size = layer['poolSize']
      self.padding = 0
      self.stride = layer['stride']
      self.dict.update({'pool_size':layer['poolSize'], 'stride':layer['stride'],
      'start':layer['start']})
    elif self.name in ['rnorm', 'cmrnorm']:
      self.filter_size = layer['size']
      self.stride = 1
      self.padding = 0
      self.dict.update({'pow':layer['pow'], 'size':layer['size'], 'scale':layer['scale']})
    elif self.name == 'neuron':
      pass
    else:
      assert False, 'Type Error for ConvRequest %s' % self.name

  def define_request(self):
    self.type = 'share'
          
    if self.state == sisw:
      self.list.append(self.dict)
 
    elif self.state == disw_b:
      self.define_disw_b()

    elif self.state == sidw:
      if self.name == 'conv':
        num_filter = divup(self.num_filter, self.num_worker)
        last_num_filter = self.num_filter - num_filter * (self.num_worker - 1)
        dict = copy.deepcopy(self.dict)
        dict['filter_shape'] = self.weight_shape[:-1] + (num_filter,)
        self.list.append(dict)

        if num_filter != last_num_filter:
          self.type = 'max'
          dict2 = copy.deepcopy(dict)
          dict2['filter_shape'] = self.filter_shape[:-1] + (last_num_filter, )
          self.list.append(dict2)
    
    elif self.state == disw_i:
      dic_set = set()
      for i in range(self.num_worker):
        input_varray = VArray(self.input_shape, self.num_worker, i)
        output_varray = VArray(self.output_shape, self.num_worker, i)
        output_shape = output_varray.local_shape
        input_shape = input_varray.local_shape
        if self.name != 'neuron':
          if self.name == 'pool':
            num_output = tuple(output_shape[1:3])
          else:
            num_output = None
          input_shape = input_varray.cross_communicate(self.stride, self.filter_size, -self.padding, num_output)[0]
        dic_set.add((input_shape, output_shape))

      for input_shape, output_shape in dic_set:
        dict = copy.deepcopy(self.dict)
        dict['input_shape'] = input_shape
        dict['output_shape'] = output_shape
        self.list.append(dict)
      
      if len(dic_set) != 1:
        self.type = 'max'
        
    else:
      assert False, 'Distribution Error for ' + self.name + str(self.state)



class FCRequestWriter(RequestWriter):
  def __init__(self, layer, state, num_worker):
    RequestWriter.__init__(self, layer, state, num_worker)
    if self.name == 'fc':
      self.output_size = layer['outputSize']
      self.input_size = layer['input_shape'][0]
      self.weight_shape = layer['weight_shape']

  def define_request(self):
    self.type = 'share'

    if self.state == sisw:
      self.list.append(self.dict)
 
    elif self.state == disw_b:
      self.define_disw_b()

    elif self.state == sidw_f:
      if self.name == 'fc':
        output_size = divup(self.output_size, self.num_worker)
        last_output_size = self.output_size - output_size * (self.num_worker - 1)
        dict = copy.deepcopy(self.dict)
        dict['weight_shape'] = (output_size, self.input_size)
        self.list.append(dict)

        if output_size != last_output_size:
          self.type = 'max'
          dict2 = copy.deepcopy(dict)
          dict2['weight_shape'] = (last_output_size, self.input_size)
          self.list.append(dict2)

    else:
      assert False, 'Distribution Error for ' + self.name


class RequestProxy(object):
  def __init__(self, writer):
    self.writer = writer
    self.id = 0
    print >> self.writer, '['

  def write_request(self, layer, state, num_worker, conv_end):
    if conv_end != True:
      conv = ConvRequestWriter(layer, state, num_worker)
      conv.write_request(self.id, self.writer)
    else:
      fc = FCRequestWriter(layer, state, num_worker)
      fc.write_request(self.id, self.writer)
    self.id += 1
  
  def finish(self):
    print >> self.writer, {'end':'true'}, ']'
