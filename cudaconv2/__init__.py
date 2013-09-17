from .cudaconv2 import *

def init(device=-1):
  # MAGIC MAGIC
  from pycuda import driver
  driver.init()
  
  if device == -1:
    from pycuda.tools import make_default_context
    context = make_default_context()
    device = context.get_device()
  else:
    device = driver.Device(device % driver.Device.count())
    context = device.make_context()
  
  print 'Starting up using device: %s:%s' % (device.name(), device.pci_bus_id()) 
  import atexit
  atexit.register(context.detach)

  return context
