from .caffe import *
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
