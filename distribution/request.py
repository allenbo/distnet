from distbase.util import divup, issquare
from distbase.state import sisw, sidw, disw_i, sidw_f, disw_b
import copy
import StringIO
import json
import numpy as np
from fake_varray import VArray


class Request(object):
  def __init__(self, id, name, layer_name, type, num_worker, state, list):
    self.id = id
    self.name = name
    self.type = type
    self.list = list
    self.num_worker = num_worker
    self.state = state
    self.layer_name = layer_name

  def write_request(self, fout):
    string_out = StringIO.StringIO()
    decr_dic = {'id': self.id, 'op':self.name, 'layer_name':self.layer_name, 'type':self.type, 'workers':self.num_worker,
        'state':self.state, 'tests':len(self.list)}
    item = {'decr': decr_dic, 'param': self.list}
    #json.dump(item, string_out, sort_keys = True, indent = 2, separators = (',', ': '))
    json.dump(item, string_out)
    print >> fout , string_out.getvalue(), ','

class RequestWriter(object):
  def __init__(self, layer, state, num_worker, backend):
    self.state = state
    self.num_worker = num_worker
    self.backend = backend
    self.name = layer['type']
    self.layer_name = layer['name']
    if self.backend == 'cudaconv':
      import cudaconv_backend as cm_backend
    elif self.backend == 'cudaconv3':
      import cudaconv3_backend as cm_backend
    elif self.backend == 'caffe':
      import caffe_backend as cm_backend
    else:
      assert False, 'There is not such backend %s' % (self.backend)
    self.IMAGE_BATCH = cm_backend.ConvDataLayout.BATCH
    self.IMAGE_HEIGHT = cm_backend.ConvDataLayout.HEIGHT
    self.IMAGE_WIDTH = cm_backend.ConvDataLayout.WIDTH
    self.IMAGE_CHANNEL = cm_backend.ConvDataLayout.CHANNEL
    self.FILTER_HEIGHT = cm_backend.FilterLayout.HEIGHT
    self.FILTER_WIDTH = cm_backend.FilterLayout.WIDTH
    self.FILTER_CHANNEL = cm_backend.FilterLayout.CHANNEL
    self.FILTER_NUM = cm_backend.FilterLayout.NUM
    self.FC_BATCH = 1
    self.WEIGHT_FIRST = 0

    self.input_shape = layer['input_shape']
    self.output_shape = layer['output_shape']
    self.batch_size = self.input_shape[-1]
    self.dict = {
        'input_shape':self.input_shape,
        'output_shape':self.output_shape,
        'backend':self.backend,
        }
    self.list = []

  def define_request(self):
    assert False, 'Implementation missing in RequestWriter'

  def define_disw_b(self, min_num):
    dic_set = set()
    for i in range(self.num_worker):
      batch_idx = self.FC_BATCH if len(self.input_shape) == 2 else self.IMAGE_BATCH
      input_varray = VArray(self.input_shape, self.num_worker, i, batch_idx, min_num = min_num)
      output_varray = VArray(self.output_shape, self.num_worker, i, batch_idx, min_num = min_num)
      input_shape = input_varray.local_shape
      output_shape = output_varray.local_shape

      if input_shape:
        dic_set.add((input_shape, output_shape))

    self.num_worker = input_varray.num_worker
    for input_shape, output_shape in dic_set:
      dict = copy.deepcopy(self.dict)
      dict['input_shape'] = input_shape
      dict['output_shape'] = output_shape
      self.list.append(dict)

    if len(dic_set) != 1:
      self.type = 'max'


  def write_request(self, id, fout):
    self.define_request()
    Request(id, self.name, self.layer_name, self.type, self.num_worker, self.state, self.list).write_request(fout)


class ConvRequestWriter(RequestWriter):
  ''' This is not just for conv layer, it should work for conv stack'''
  def __init__(self, layer, state, num_worker, backend):
    RequestWriter.__init__(self, layer, state, num_worker, backend)
    if self.name == 'conv':
      self.padding, self.filter_size, self.stride = layer['padding'], layer['filterSize'], layer['stride']
      self.num_filter = layer['numFilter']
      self.weight_shape = layer['weight_shape']
      self.dict['filter_shape'] = self.weight_shape
      self.dict['stride'] = self.stride
      self.dict['padding'] = self.padding
      if 'sumWidth' in layer:
        self.dict['weight_sum'] = layer['sumWidth']
      elif 'partialSum' in layer:
        self.dict['weight_sum'] = layer['partialSum']
      else:
        self.dict['weight_sum'] = -1
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
      self.define_disw_b(1)

    elif self.state == sidw:
      dic_set = set()
      min_num = 1
      if self.name == 'cmrnorm':
        min_num = self.dict['size']
      for i in range(self.num_worker):
        output_varray = VArray(self.output_shape, self.num_worker, i, self.IMAGE_CHANNEL, min_num = min_num)
        output_shape = output_varray.local_shape
        if self.name == 'conv':
          filter_varray = VArray(self.weight_shape, self.num_worker, i, self.FILTER_NUM, min_num = min_num)
          filter_shape = filter_varray.local_shape
          if filter_shape:
            dic_set.add((filter_shape, output_shape))
        else:
          input_varray = VArray(self.input_shape, self.num_worker, i, self.IMAGE_CHANNEL, min_num = min_num)
          input_shape = input_varray.local_shape
          if input_shape:
            dic_set.add((input_shape, output_shape))

      self.num_worker = output_varray.num_worker
      for shape, output_shape in dic_set:
        dict = copy.deepcopy(self.dict)
        dict['filter_shape' if self.name == 'conv' else 'input_shape'] = shape
        dict['output_shape'] = output_shape
        self.list.append(dict)

      if len(dic_set) != 1:
        self.type = 'max'

    elif self.state == disw_i:
      overlapping_max =  0
      actual_data_max =  0
      dic_set = set()
      for i in range(self.num_worker):
        if issquare(self.num_worker):
          input_varray = VArray(self.input_shape, self.num_worker, i, (self.IMAGE_HEIGHT, self.IMAGE_WIDTH))
          output_varray = VArray(self.output_shape, self.num_worker, i, (self.IMAGE_HEIGHT, self.IMAGE_WIDTH))
        else:
          input_varray = VArray(self.input_shape, self.num_worker, i, self.IMAGE_HEIGHT)
          output_varray = VArray(self.output_shape, self.num_worker, i, self.IMAGE_HEIGHT)
        output_shape = output_varray.local_shape
        input_shape = input_varray.local_shape
        if self.name not in ['neuron', 'cmrnorm']:
          input_shape, actual_data, overlapping = input_varray.image_communicate((self.IMAGE_HEIGHT, self.IMAGE_WIDTH), self.stride, self.filter_size, -self.padding, output_area = output_varray.local_area)

          if self.name in ['rnorm']:
            output_shape = input_shape

          if input_shape:
            overlapping_max = max(overlapping, overlapping_max)
            actual_data_max = max(actual_data, actual_data_max)

        if input_shape:
          dic_set.add((input_shape, output_shape))

      self.num_worker = output_varray.num_worker
      self.dict['overlapping'] = overlapping_max
      self.dict['actual_data'] = actual_data_max
      for input_shape, output_shape in dic_set:
        dict = copy.deepcopy(self.dict)
        dict['input_shape'] = input_shape
        dict['output_shape'] = output_shape
        dict['padding'] = 0
        self.list.append(dict)

      if len(dic_set) != 1:
        self.type = 'max'
    else:
      assert False, 'Distribution Error for ' + self.name + str(self.state)



class FCRequestWriter(RequestWriter):
  def __init__(self, layer, state, num_worker, backend):
    RequestWriter.__init__(self, layer, state, num_worker, backend)
    if self.name == 'fc':
      self.output_size = layer['outputSize']
      self.input_size = layer['input_shape'][0]
      self.weight_shape = layer['weight_shape']
      self.drop_out = layer.get('dropRate', 0)
      self.dict.update({'weight_shape': self.weight_shape, 'drop_out':self.drop_out})

  def define_request(self):
    self.type = 'share'

    if self.state == sisw:
      self.list.append(self.dict)

    elif self.state == disw_b:
      self.define_disw_b(8)

    elif self.state == sidw_f:
      dic_set = set()

      if self.name in ['fc', 'neuron']:
        for i in range(self.num_worker):
          output_varray = VArray(self.output_shape, self.num_worker, i, self.WEIGHT_FIRST)
          output_shape = output_varray.local_shape
          if self.name == 'fc':
            weight_varray = VArray(self.weight_shape, self.num_worker, i, self.WEIGHT_FIRST)
            weight_shape = weight_varray.local_shape
            if weight_shape:
              dic_set.add((weight_shape, output_shape))
          else:
            input_varray = VArray(self.input_shape, self.num_worker, i, self.WEIGHT_FIRST)
            input_shape = input_varray.local_shape
            if input_shape:
              dic_set.add((input_shape, output_shape))

        for shape, output_shape in dic_set:
          dict = copy.deepcopy(self.dict)
          dict['weight_shape' if self.name == 'fc' else 'input_shape'] = shape
          dict['output_shape'] = output_shape
          self.list.append(dict)

        if len(dic_set) != 1:
          self.type = 'max'

    else:
      assert False, 'Distribution Error for ' + self.name


class RequestProxy(object):
  def __init__(self, writer, backend):
    self.writer = writer
    self.id = 0
    self.backend = backend
    print >> self.writer, '['

  def write_request(self, layer, state, num_worker, conv_end):
    if conv_end != True:
      conv = ConvRequestWriter(layer, state, num_worker, self.backend)
      conv.write_request(self.id, self.writer)
    else:
      fc = FCRequestWriter(layer, state, num_worker, self.backend)
      fc.write_request(self.id, self.writer)
    self.id += 1

  def finish(self):
    print >> self.writer, "{\"end\":\"true\"}]"
