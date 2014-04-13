from new_gpuarray import *
from backend import *
from operation import *

# The order of imports is important, since the operation and new_gpuarray will change the behavior
# of some functio in aux_operation

def get_seed():
  import time
  return int(time.time())
