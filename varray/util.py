import time
import threading
import inspect
import sys
import os
import math
import os.path

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

def log(fmt, *args, **kw):
  with log_lock:
    level = kw.get('level', 0)
    level = log_level[level]
    caller_frame = inspect.stack()[1]
    filename = os.path.basename(caller_frame[1])
    lineno = caller_frame[2]
    print >> sys.stderr, "==%s== %.4fs %s:%s ::" % (level, time.time() - start, filename, lineno) ,
    print >> sys.stderr, "%s" % (fmt % args)


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
