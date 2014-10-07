import garray import ConvDataLayout, FCDataLayout
import garray
from ndarray import allocate, size, array
from context import CONTEXTMANAGER
from distbase import util
import re

class Para(object):
  NumWorker = size
  def __init__(self):
    pass
  def init_output(self, shape):
    pass
  def init_weight(self, shape = None, array = None):
    pass
  def init_bias(self, shape = None, array = None):
    pass
  def get_allocator(self):
    return self

  @staticmethod
  def get(desc):
    if desc == '':
      para = FakePara()
      return para
    elif desc.startswith('R'):
      match = re.search(r'R\[(\d+)\]')
      if match:
        N = match.group(1)
        assert N == Para.NumWorker
        para =  ReplicaPara(N)
        return para
      else:
        return None
    elif desc.startswith('M'):
      match = re.search(r'M\[(\d+)\]')
      if match:
        N = match.group(1)
        assert N == Para.NumWorker
        para = ConvModelPara(N)
        return para
      else:
        return None
    elif desc.startswith('BI'):
      match = re.search(r'BI\[(\d+):(\d+)\]')
      if match:
        N = match.group(1)
        M = match.group(2)
        assert N*M == NumWorker
        para = BatchImagePara(N, M)
        return para
      else:
        return None
    elif desc.startswith('I'):
      match = re.search(r'I\[(\d+)\]')
      if match:
        N = match.group(1)
        assert N == Para.NumWorker
        para = ImagePara(N)
        return para
      else:
        return None
    elif desc.startswith("BB"):
      match = re.search(r'BB\[(\d+):(\d+)\]')
      if match:
        N = match.group(1)
        M = match.group(2)
        assert N*M == NumWorker
        para = BatchBatchPara(N, M)
        return para
      else:
        return None
    elif desc.startswith('B'):
      match = re.search(r'B\[(\d+)\]')
      if match:
        N = match.group(1)
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
        FM[N] means FCModelPara with N worker
        CM[N] means ConvModelPara with N worker
        R[N] means ReplicaPara with N worker
        empty string means FakePara, single GPU will use this
    '''
    para = Para.get(desc)
    if not para :
      util.log_fatal('Incorrect definition on parallelism ' + desc)
    else:
      return para

class FakePara(Para):
  def init_output(self, shape = None, array = None):
    if array:
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
  def __init__(self):
    pass

  def init_weight(self, shape = None, array = None):
    if array:
      return array(array, context = self._context)
    else:
      assert shape is not None
      return allocate(shape = shape, context = self._context)

  init_bias = init_weight

class BatchPara(DataPara):
  def __init__(self, num_worker, fc = False):
    assert num_worker > 1
    self._num_worker = num_worker
    self._context = CONTEXTMANAGER.build_context([1] * self._num_worker)

  def init_ouptut(self, shape):
    return allocate(shape = shape, global_slice_dim = self._global_dim, context = self._context)

  def to_fc(self):
    self._global_dim = FCDataLayout.BATCH
    return self

  def to_conv(self):
    self._global_dim = ConvDataLayout.BATCH
    return self

class BatchBatchPara(DataPara):
  def __init__(self, num_group, num_worker_per_group):
    assert num_group > 1 and num_worker_per_group > 1
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

class ImagePara(DataPara):
  def __init__(self, num_worker):
    assert num_worker > 1
    self._num_worker = num_worker
    self._context = CONTEXTMANAGER.build_context([self._num_worker])

  def init_output(self, shape):
    if  util.issqrt(self._num_worker):
      group_slice_dim = (ConvDataLayout.HEIGHT, ConvDataLayout.WIDTH)
    else:
      group_slice_dim = ConvDataLayout.HEIGHT
    return allocate(shape = shape, group_slice_dim = group_slice_dim, context = self._context)

  def to_fc(self):
    assert  False

  def to_conv(self):
    return self

class BatchImagePara(DataPara):
  def __init__(self, num_group, num_worker_per_group):
    assert num_group > 1 and num_worker_per_group > 1
    self._num_group  = num_group
    self._num_worker_per_group = num_worker_per_group
    self._context = CONTEXTMANAGER.build_context([self._num_worker_per_group] * self._num_group)

  def init_output(self, shape):
    if  util.issqrt(self._num_worker_per_group):
      group_slice_dim = (ConvDataLayout.HEIGHT, ConvDataLayout.WIDTH)
    else:
      group_slice_dim = ConvDataLayout.HEIGHT

    return allocate(shape = shape, global_slice_dim = ConvDataLayout.BATCH, group_slice_dim =
            group_slice_dim, context = self._context)

  def to_fc(self):
    assert  False

  def to_conv(self):
    return self

class ModelPara(Para):
  def __init__(self, num_worker)
    assert num_worker > 1
    self._num_worker = num_worker
    self._context = CONTEXTMANAGER.build_context([self._num_worker])

  def init_bias(self, shape = None, array = None):
    assert len(shape) == 2 and shape[1] = 1
    if array:
      return array(array, group_slice_dim = 0, context = self._context)
    else:
      assert shape is not None
      return allocate(shape = shape, group_slice_dim = 0, context = self._context)


  def init_output(self, shape):
    return allocate(shape = shape, group_slice_dim = self._output_group_dim, context = self._contenxt)

  def init_weight(self, shape = None, array = None):
    if array:
      return array(array, group_slice_dim = self._weight_group_dim, context = self._context)
    else:
      assert shape is not None:
      return allocate(shape = shape, group_slice_dim =self._weight_group_dim, context = self._context)

  def to_fc(self):
    self._output_group_dim = FCDataLayout.NEURON
    self._weight_group_dim = WeightLayout.OUTPUT
    return self

  def to_conv(self):
    self._output_group_dim = ConvDataLayout.CHANNEL
    self._weight_group_dim = FilterLayout.NUM
    return self

class ReplicaPara(Para):
  def __init__(self, num_worker):
    assert num_worker > 1
    self._num_worker = num_worker
    self._cotext = CONTEXTMANAGER.build_context([self._num_worker])

  def init_output(self, shape = None, array = None):
    if array:
      return array(array, context = self._context)
    else:
      assert shape is not None
      return allocate(shape = shape, context = self._context)

  init_weight = init_bias = init_output

  def to_fc(self): return self
  def to_conv(self): return self
