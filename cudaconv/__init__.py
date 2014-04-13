from .cudaconv2 import *


class ConvDataLayout(object):
  CHANNEL = 0
  HEIGHT = 1
  WIDTH = 2
  BATCH = 3
  DIM = 4

  @staticmethod
  def get_output_shape(image_y, image_x, channel, batch_size):
    return (channel, image_y, image_x, batch_size)
  

class FilterLayout(object):
  CHANNEL = 0
  HEIGHT = 1
  WIDTH = 2
  NUM = 3
  DIM = 4

  @staticmethod
  def get_filter_shape(filter_size, channel, num):
    return (channel, filter_size, filter_size, num)

class FCDataLayout(object):
  NEURON = 0
  BATCH = 1
  DIM = 2

  @staticmethod
  def get_output_shape(neuron, batch_size):
    return (neuron, batch_size)

class WeightLayout(object):
  OUTPUT = 0
  INPUT = 1
  DIM = 2

  @staticmethod
  def get_weight_shape(input, output):
    return (output, input)

backend = 'cudaconv'

CONTEXT = None
def init(device=-1):
  global CONTEXT
  if CONTEXT is not None:
    return
  
  # MAGIC MAGIC
  from pycuda import driver
  driver.init()
  
  if device == -1:
    from pycuda.tools import make_default_context
    CONTEXT = make_default_context()
    device = CONTEXT.get_device()
  else:
    device = driver.Device(device % driver.Device.count())
    CONTEXT = device.make_context()
  
  print 'Starting up using device: %s:%s' % (device.name(), device.pci_bus_id()) 
  import atexit
  atexit.register(CONTEXT.detach)
  return CONTEXT
