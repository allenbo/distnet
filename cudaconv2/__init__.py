from .cudaconv2 import *

CONTEXT = None
def init():
  global CONTEXT

  if CONTEXT is not None:
    return CONTEXT

  # MAGIC MAGIC
  from pycuda import driver
  driver.init()
  from pycuda.tools import make_default_context
  context = make_default_context()
  device = context.get_device()
  import atexit
  atexit.register(context.detach)

  CONTEXT = context
  return CONTEXT
