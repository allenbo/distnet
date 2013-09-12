import cPickle
import os
import sys
import threading
import time
import traceback
import numpy as np

program_start = time.time()
log_mutex = threading.Lock()
def log(msg, *args, **kw):
  with log_mutex:
    caller = sys._getframe(1)
    filename = caller.f_code.co_filename
    lineno = caller.f_lineno
    now = time.time() - program_start
    if 'exc_info' in kw:
      exc = ''.join(traceback.format_exc())
    else:
      exc = None
    print >> sys.stderr, '%.3f:%s:%d: %s' % (now, os.path.basename(filename), lineno, msg % args)
    if exc:
      print >> sys.stderr, exc


class Timer:
  def __init__(self):
    self.func_time = {}
    self.last_time = 0.0

  def start(self):
    self.last_time = time.time()

  def end(self, func_name):
    ftime = time.time() - self.last_time
    if func_name in self.func_time:
      self.func_time[func_name] += ftime
    else:
      self.func_time[func_name] = ftime

  def report(self):
    dic = self.func_time
    for key in sorted(dic):
      print key, ':', dic[key]
timer = Timer()


class EZTimer(object):
  def __init__(self, msg=''):
    self.msg = msg
    self.start = time.time()
    
  def __del__(self):
    log('Operation %s finished in %.5f seconds', self.msg, time.time() - self.start) 
    

def divup(x, base):
  if x / base * base == x:
    return int(x / base)
  else:
    return int(x / base + 1)

def load(filename):
  with open(filename, 'rb') as f:
    model = cPickle.load(f)
  return model

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def isinteger(value):
  try:
    int(value)
    return True
  except ValueError:
    return False

def string_to_int_list(str):
  if str is None: return []
  str = str.strip()
  if str.find('-') != -1:
    f = int(str[0:str.find('-')])
    t = int(str[str.find('-') + 1:])

    return range(f, t + 1)
  elif str.startswith('['):
    str = str[1:-1]
    return [int(s) for s in str.split(', ')]
  else:
    elt = int(str)
    return [elt]

def string_to_float_list(str):
  if str is None: return []
  str = str.strip()

  if str.startswith('['):
    str = str[1:-1]
    return [float(s) for s in str.split(', ')]
  else:
    return [float(str)]

def print_matrix(x, name, row_from = 0, row_to = 0, col_from = 0, col_to = 0):
  from pycuda import gpuarray
  print name
  if row_to == 0:
    row_to = 10 #x.shape[0]
  if col_to == 0:
    col_to = 1 #x.shape[1]
  if isinstance(x, gpuarray.GPUArray):
    a = x.get()[row_from: row_to , col_from: col_to]
  else:
    a = x[row_from: row_to , col_from: col_to]

  for rows in a:
    for i in rows:
      print '%.15f' % i,
    print ''

def abs_mean(x):
  from pycuda import gpuarray
  if isinstance(x, gpuarray.GPUArray):
    return (gpuarray.sum(x.__abs__()) / x.size).get().item()
  if isinstance(x, np.ndarray):
    return np.mean(np.abs(x))
  

class Assert(object):
  @staticmethod
  def all_eq(a, b):
    import numpy
    if hasattr(a, 'shape') and hasattr(b, 'shape'):
      assert a.shape == b.shape, 'Mismatched shapes: %s %s' % (a.shape, b.shape)
      
    assert numpy.all(a == b), 'Failed: \n%s\n ==\n%s' % (a, b)
  
  @staticmethod
  def eq(a, b): assert (a == b), 'Failed: %s == %s' % (a, b)
  
  @staticmethod
  def ne(a, b): assert (a == b), 'Failed: %s != %s' % (a, b)
  
  @staticmethod
  def gt(a, b): assert (a > b), 'Failed: %s > %s' % (a, b)
  
  @staticmethod
  def lt(a, b): assert (a < b), 'Failed: %s < %s' % (a, b)
  
  @staticmethod
  def ge(a, b): assert (a >= b), 'Failed: %s >= %s' % (a, b)
  
  @staticmethod
  def le(a, b): assert (a <= b), 'Failed: %s <= %s' % (a, b)
  
  @staticmethod
  def true(expr): assert expr, 'Failed: %s == True' % (expr)
  
  @staticmethod
  def isinstance(expr, klass): 
    assert isinstance(expr, klass), 'Failed: isinstance(%s, %s) [type = %s]' % (expr, klass, type(expr))
  
  @staticmethod
  def no_duplicates(collection):
    d = collections.defaultdict(int)
    for item in collection:
      d[item] += 1
    
    bad = [(k,v) for k, v in d.iteritems() if v > 1]
    assert len(bad) == 0, 'Duplicates found: %s' % bad
  
