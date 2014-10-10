from garray import ConvDataLayout, FCDataLayout, WeightLayout, FilterLayout
import garray
import ndarray
from ndarray import allocate, size
from context import CONTEXTMANAGER
from distbase import util
import re
import numpy as np

class Para(object):
  NumWorker = size
  def __init__(self, name):
    self.name = name
  def init_output(self, shape):
    pass
  def init_weight(self, shape = None, array = None):
    pass
  def init_bias(self, shape = None, array = None):
    pass
  def before_fprop(self, layer):
    pass
  def after_bprop(self, layer):
    pass
  def after_weight(self, layer):
    pass

  @staticmethod
  def get(desc):
    if desc == '' or Para.NumWorker == 1:
      para = FakePara()
      return para
    elif desc.startswith('R'):
      match = re.search(r'R\[(\d+)\]', desc)
      if match:
        N = int(match.group(1))
        assert N == Para.NumWorker
        para =  ReplicaPara(N)
        return para
      else:
        return None
    elif desc.startswith('M'):
      match = re.search(r'M\[(\d+)\]', desc)
      if match:
        N = int(match.group(1))
        assert N == Para.NumWorker
        para = ModelPara(N)
        return para
      else:
        return None
    elif desc.startswith('BI'):
      match = re.search(r'BI\[(\d+):(\d+)\]', desc)
      if match:
        N = int(match.group(1))
        M = int(match.group(2))
        assert N*M == NumWorker
        para = BatchImagePara(N, M)
        return para
      else:
        return None
    elif desc.startswith('I'):
      match = re.search(r'I\[(\d+)\]', desc)
      if match:
        N = int(match.group(1))
        assert N == Para.NumWorker
        para = ImagePara(N)
        return para
      else:
        return None
    elif desc.startswith("BB"):
      match = re.search(r'BB\[(\d+):(\d+)\]', desc)
      if match:
        N = int(match.group(1))
        M = int(match.group(2))
        assert N*M == NumWorker
        para = BatchBatchPara(N, M)
        return para
      else:
        return None
    elif desc.startswith('B'):
      match = re.search(r'B\[(\d+)\]', desc)
      if match:
        N = int(match.group(1))
        assert N == Para.NumWorker
        para = BatchPara(N)
        return para
      else:
        return None
    else:
      return None

  @staticmethod
  def build_para_adapter(desc):
    ''' B[N] means BatchPara with N workers
        BB[N:M] means BatchBatchPara with N groups, M workers per group
        I[N] means ImagePara with N workers
        BI[N:M] means BatchImagePara with N groups, M workers per group
        M[N] means ConvModelPara with N worker
        R[N] means ReplicaPara with N worker
        empty string means FakePara, single GPU will use this
    '''
    para = Para.get(desc)
    if not para :
      util.log_fatal('Incorrect definition on parallelism ' + desc)
    else:
      return para

class FakePara(Para):
  def __init__(self):
    Para.__init__(self, 'F')

  def init_output(self, shape = None, array = None):
    if array is not None:
      return garray.array(array, dtype = np.float32)
    else:
      assert shape is not None
      return garray.GPUArray(shape, dtype = np.float32)

  init_bias = init_weight = init_output

  def random_uniform(self, shape):
    return garray.random_uniform(shape)

  def to_fc(self):
    return self

  def to_conv(self):
    return self

class DataPara(Para):
  def __init__(self, name):
    Para.__init__(self, name)

  def init_weight(self, shape = None, array = None):
    if array is not None:
      return ndarray.array(array, context = self._context)
    else:
      assert shape is not None
      return allocate(shape = shape, context = self._context)

  init_bias = init_weight

  def after_weight(self, layer):
    weight_grad = layer.weight.grad
    weight_grad.write(area = weight_grad.global_area, data = weight_grad.DATA, propagate = True)
    bias_grad = layer.bias.grad
    bias_grad.write(area = bias_grad.global_area, data = bias_grad.DATA, propagate = True)

class BatchPara(DataPara):
  def __init__(self, num_worker):
    assert num_worker > 1
    DataPara.__init__(self, 'B')
    self._num_worker = num_worker
    self._context = CONTEXTMANAGER.build_context([self._num_worker])

  def init_output(self, shape):
    return allocate(shape = shape, group_slice_dim = self._group_dim, context = self._context)

  def to_fc(self):
    self._group_dim = FCDataLayout.BATCH
    return self

  def to_conv(self):
    self._group_dim = ConvDataLayout.BATCH
    return self

  def before_fprop(self, layer):
    input = layer._prev_layer.output
    prev_conv = False if not hasattr(layer, 'prev_conv') else layer.prev_conv
    input.batch_communicate(input.group_rank, self._group_dim if prev_conv == False else ConvDataLayout.BATCH)

  def after_bprop(self, layer):
    outgrad = layer._prev_layer.output_grad
    input = layer._prev_layer.output
    if not outgrad.has_local_cache(): return
    outgrad.write(area = input.local_cache_area, data = outgrad.local_cache, propagate = True)

class BatchBatchPara(DataPara):
  def __init__(self, num_group, num_worker_per_group):
    assert num_group > 1 and num_worker_per_group > 1
    DataPara.__init__(self, 'BB')
    self._num_group = num_group
    self._num_worker_per_group = num_worker_per_group
    self._context = CONTEXTMANAGER.build_context([self._num_worker_per_group] * self._num_group)

  def init_output(self, shape):
    return allocate(shape = shape, global_slice_dim = self._global_dim,
            group_slice_dim = self._group_dim, context = self._context)

  def to_fc(self):
    self._global_dim = FCDataLayout.BATCH
    self._group_dim = FCDataLayout.BATCH
    return self

  def to_conv(self):
    self._global_dim = ConvDataLayout.BATCH
    self._group_dim = ConvDataLayout.BATCH
    return self

  before_fprop = BatchPara.before_fprop
  after_bprop = BatchPara.after_bprop

class ImagePara(DataPara):
  def __init__(self, num_worker):
    assert num_worker > 1
    DataPara.__init__(self, 'I')
    self._num_worker = num_worker
    self._context = CONTEXTMANAGER.build_context([self._num_worker])

  def init_output(self, shape):
    if  util.issquare(self._num_worker):
      group_slice_dim = (ConvDataLayout.HEIGHT, ConvDataLayout.WIDTH)
    else:
      group_slice_dim = ConvDataLayout.HEIGHT
    return allocate(shape = shape, group_slice_dim = group_slice_dim, context = self._context)

  def to_fc(self):
    assert  False

  def to_conv(self):
    return self

  def before_fprop(self, layer):
    input = layer._prev_layer.output
    r, c = ConvDataLayout.HEIGHT, ConvDataLayout.WIDTH
    output_area = layer.output.local_area
    #padding is a little trickly, in ImagePara, padding should be changed to be 0, but other paras
    #don't have to do it. So we have to save padding in para object.
    padding = 0
    if hasattr(self, 'padding'):
      padding = self.padding
      layer.padding = 0
    elif hasattr(layer, 'padding'):
      padding = -layer.padding
      self.padding = padding
      layer.padding = 0
    stride = layer.stride if hasattr(layer, 'stride') else 1
    if layer.type == 'conv':
      filter_size = layer.filterSize
    elif layer.type == 'pool':
      filter_size = layer.poolSize
    elif layer.type in ['rnorm', 'cmrnorm']:
      filter_size = 0

    input.image_communicate(slice_dim = (r, c), padding = padding, stride = stride, filter_size = filter_size, output_area = output_area)

  def after_bprop(self, layer):
    r, c = ConvDataLayout.HEIGHT, ConvDataLayout.WIDTH
    input = layer._prev_layer.output
    out_grad = layer._prev_layer.output_grad
    if layer.type == 'conv' and self.padding != 0:
      out_grad.local_cache = input.unpad(data = out_grad.local_cache,
                               padding = self.padding,
                               old_shape = out_grad.local_cache.shape,
                               old_area = input.local_cache_area,
                               slice_dim = (r, c))
      out_grad.write(area = input.local_cache_area, data = out_grad.local_cache, propagate = True)
      del out_grad.local_cache
    else:
      out_grad.write(area = input.local_cache_area, data = out_grad.local_cache, propagate = True)


class BatchImagePara(DataPara):
  def __init__(self, num_group, num_worker_per_group):
    assert num_group > 1 and num_worker_per_group > 1
    DataPara.__init__(self, 'BI')
    self._num_group  = num_group
    self._num_worker_per_group = num_worker_per_group
    self._context = CONTEXTMANAGER.build_context([self._num_worker_per_group] * self._num_group)

  def init_output(self, shape):
    if  util.issquare(self._num_worker_per_group):
      group_slice_dim = (ConvDataLayout.HEIGHT, ConvDataLayout.WIDTH)
    else:
      group_slice_dim = ConvDataLayout.HEIGHT

    return allocate(shape = shape, global_slice_dim = ConvDataLayout.BATCH, group_slice_dim =
            group_slice_dim, context = self._context)

  def to_fc(self):
    assert  False

  def to_conv(self):
    return self

  before_fprop = ImagePara.before_fprop
  afterr_bprop = ImagePara.after_bprop

class ModelPara(Para):
  def __init__(self, num_worker):
    assert num_worker > 1
    Para.__init__(self, 'M')
    self._num_worker = num_worker
    self._context = CONTEXTMANAGER.build_context([self._num_worker])

  def init_bias(self, shape = None, array = None):
    if array is not None:
      return ndarray.array(array, group_slice_dim = 0, context = self._context)
    else:
      assert shape is not None
      assert len(shape) == 2 and shape[1] == 1
      return allocate(shape = shape, group_slice_dim = 0, context = self._context)


  def init_output(self, shape):
    return allocate(shape = shape, group_slice_dim = self._output_group_dim, context = self._context)

  def init_weight(self, shape = None, array = None):
    if array is not None:
      return ndarray.array(array, group_slice_dim = self._weight_group_dim, context = self._context)
    else:
      assert shape is not None
      return allocate(shape = shape, group_slice_dim =self._weight_group_dim, context = self._context)

  def to_fc(self):
    self._output_group_dim = FCDataLayout.NEURON
    self._weight_group_dim = WeightLayout.OUTPUT
    return self

  def to_conv(self):
    self._output_group_dim = ConvDataLayout.CHANNEL
    self._weight_group_dim = FilterLayout.NUM
    return self

  def before_fprop(self, layer):
    if layer.type in ['fc', 'conv']: # weighted layer
      layer._prev_layer.output.global_communicate()
    else:
      input = layer._prev_layer.output
      layer._prev_layer.output.channel_communicate(input.group_rank, ConvDataLayout.CHANNEL)

  def after_bprop(self, layer):
    input = layer._prev_layer.output
    out_grad = layer._prev_layer.output_grad
    out_grad.write(area = out_grad.global_area, data = out_grad.local_cache, propagate = True)

  def random_uniform(self, shape):
    #only FC layer needs random uniform, so same distribution with output
    return ndarray.random_uniform(shape, group_slice_dim = self._output_group_dim, context = self._context)

class ReplicaPara(Para):
  def __init__(self, num_worker):
    assert num_worker > 1
    Para.__init__(self, 'R')
    self._num_worker = num_worker
    self._context = CONTEXTMANAGER.build_context([self._num_worker])

  def init_output(self, shape = None, array = None):
    if array is not None:
      return ndarray.array(array, context = self._context)
    else:
      assert shape is not None
      return allocate(shape = shape, context = self._context)

  init_weight = init_bias = init_output

  def to_fc(self): return self
  def to_conv(self): return self

  def before_fprop(self, layer):
    layer._prev_layer.output.global_communicate()

  def after_bprop(self, layer):
    input = layer._prev_layer.output
    out_grad = layer._prev_layer.output_grad
    out_grad.write(area = out_grad.global_area, data = out_grad.local_cache, propagate = False)

  def random_uniform(self, shape):
    return ndarray.random_uniform(shape = shape, context = self._context)
