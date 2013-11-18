from ndarray import *
from util import *
from area import *
from random import *
from operations import *


def distlog(_fn):
  if rank == MASTER:
    return _fn
  else:
    return lambda msg, *arg, **kw : 0
