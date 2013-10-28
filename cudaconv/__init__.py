from .cudaconv2 import *

# driver.init()
# device_info = (0, 0)
# for i in range(driver.Device.count()):
#  dev = driver.Device(i)
#  ctx = dev.make_context()
#  ctx.push()
#  free, total = driver.mem_get_info()
#  print 'Free Memory for Device', i, 'is', free / 1000000, 'MB'
#
#  if device_info[1] < free:
#    device_info = (i, free)
#
#  ctx.pop()
#  ctx.detach()

# print 'Choose Device', device_info[0]
# dev = driver.Device(device_info[0])

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
