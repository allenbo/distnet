import time
import threading
import sys
import os
import math
import os.path
import ctypes
import sys
import numpy as np

start = time.time()
log_lock = threading.Lock()

DEBUG = 0
INFO = 1
WARN = 2
ERROR = 3
FATAL = 4

log_level = { DEBUG:'DEBUG',
              INFO:'INFO',
              WARN:'WARN',
              ERROR:'ERROR',
              FATAL:'FATAL'
              }

from mpi4py import MPI
rank = MPI.COMM_WORLD.Get_rank()

def log(fmt, *args, **kw):
  with log_lock:
    level = kw.get('level', 0)
    level = log_level[level]
    caller_frame = sys._getframe(1)
    filename = os.path.basename(caller_frame.f_code.co_filename)
    lineno = caller_frame.f_lineno
    msg = "==%s== %.4fs %s:%s %s:: %s" % (level, time.time() - start, filename, lineno, rank, fmt % args)
    print >> sys.stderr, msg


def D(fmt, *args): log(fmt, *args, level=DEBUG)
def I(fmt, *args): log(fmt, *args, level=INFO)
def W(fmt, *args): log(fmt, *args, level=WARN)
def E(fmt, *args): log(fmt, *args, level=ERROR)
def F(fmt, *args): log(fmt, *args, level=FATAL)

def issquare(x):
  a = int(math.sqrt(x))
  if a ** 2 == x:
    return True
  else:
    return False

def divup(x, a):
  return x/a if x % a == 0 else x/a +1

class Timer(object):
  def __init__(self):
    self.funcs = {}
    self.start = 0

  def start(self):
    self.start = time.time()

  def end(self, name):
    if not name in self.funcs:
      self.funcs[name] = 0
    self.funcs[name] += time.time() - self.start


timer = Timer()

def timeit(fn):
  def _fn(*args, **kw):
    timer.start()
    rst = fn(*args, **kw)
    time.end(fn.__name__)
    return rst

  return _fn

