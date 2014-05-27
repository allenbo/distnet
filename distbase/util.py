import cPickle
import os
import sys
import threading
import time
import traceback
import math
import numpy as np
import warnings
import functools

#if os.environ.get('MULTIGPU', 'no') == 'yes':
#  from varray import distlog
#else:
distlog = lambda(_fn): _fn

DEBUG = 0
INFO = 1
WARN = 2
ERROR = 3
FATAL = 4

level_to_char = { DEBUG : 'D',
                  INFO : 'I',
                  WARN : 'W',
                  ERROR : 'E',
                  FATAL : 'F', 
                  }

program_start = time.time()
log_mutex = threading.Lock()

@distlog
def log(msg, *args, **kw):
  level = kw.get('level', INFO)
  with log_mutex:
    caller = sys._getframe(kw.get('caller_frame', 1))
    filename = caller.f_code.co_filename
    lineno = caller.f_lineno
    now = time.time() - program_start
    if 'exc_info' in kw:
      exc = ''.join(traceback.format_exc())
    else:
      exc = None
    print >> sys.stderr, '%s %.3f:%s:%d: %s' % (level_to_char[level], now, os.path.basename(filename), lineno, msg % args)
    if exc:
      print >> sys.stderr, exc

def log_debug(msg, *args, **kw): log(msg, *args, level=DEBUG, caller_frame=2)
def log_info(msg, *args, **kw): log(msg, *args, level=INFO, caller_frame=2)
def log_warn(msg, *args, **kw): log(msg, *args, level=WARN, caller_frame=2)
def log_error(msg, *args, **kw): log(msg, *args, level=ERROR, caller_frame=2)
def log_fatal(msg, *args, **kw): log(msg, *args, level=FATAL, caller_frame=2)




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

  def dump(self, filename):
    dic = self.func_time
    keys  = sorted(dic, key = dic.get)
    with open(filename, 'w') as f:
      for key in keys:
        print >> f, key, dic[key]


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

def issquare(x):
  a = math.sqrt(x) 
  b = int(a)
  return a == b

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
  

from types import FunctionType, CodeType

def make_copy(f, name):
  func_code = f.func_code
  new_code  = CodeType(
            func_code.co_argcount, func_code.co_nlocals, func_code.co_stacksize,
            func_code.co_flags, func_code.co_code, func_code.co_consts,
            func_code.co_names, func_code.co_varnames, func_code.co_filename,
            name, func_code.co_firstlineno, func_code.co_lnotab,
            func_code.co_freevars, func_code.co_cellvars)
  wrapper = FunctionType(
            new_code, f.func_globals, name, f.func_defaults,
            f.func_closure)
  return wrapper


def lazyinit(initializer_fn):
  '''
  (Lazily) call initializer_fn prior to invocation.
  '''
  def wrap(fn):
    def _fn(*args, **kw):
      initializer_fn()
      return fn(*args, **kw)
    return make_copy(_fn, fn.__name__)
  
  return make_copy(wrap, initializer_fn.__name__)

def timed_fn(fn):
  '''
  Time the execution of ``fn``.
  '''
  def _fn(*args, **kw):
    timer.start()
    result = fn(*args, **kw) 
    timer.end(fn.__name__)
    
    return result
    
  return make_copy(_fn, fn.__name__)

def deprecated(fn):
  '''
  A decorator which can be used to mark functions as deprecated, It will result in a warning being
  emitted when the function is used. Copy from
  https://wiki.python.org/moin/PythonDecoratorLibrary#Smart_deprecation_warnings_.28with_valid_filenames.2C_line_numbers.2C_etc..29
  '''
  @functools.wraps(fn)
  def new_func(*args, **kwargs):
    warnings.warn_explicit(
        'Call to deprecated function. {}.'.format(fn.__name__),
        category = DeprecationWarning,
        filename = fn.func_code.co_filaname,
        lineno = fn.func_code.co_firstlineno + 1
        )
    return fn(*args, **kwargs)
  return new_func

PROFILER = None

import cProfile
import yappi
from mpi4py import MPI
def enable_profile():
  global PROFILER
  if PROFILER is None:
    yappi.start()
    PROFILER = 1
    #PROFILER = cProfile.Profile()
    #PROFILER.enable()

def dump_profile():
  if PROFILER is None:
    return

  yappi.get_func_stats().save('./profile.%d' % MPI.COMM_WORLD.Get_rank(), 'pstat')
  #PROFILER.dump_stats('./profile.%d' % MPI.COMM_WORLD.Get_rank())
