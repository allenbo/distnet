#!/usr/bin/env python

from setuptools import setup, Extension

import os
import sys
from distutils.spawn import find_executable
from distutils import sysconfig

def log(str):
  print >>sys.stderr, str

if find_executable('nvcc') is None:
  log('nvcc not in path; aborting')
  sys.exit(1)

log('About to build cudaconv extension.')
cmd = 'cd cudaconv && make -j8 PYTHON_INCLUDE="%s"' % sysconfig.get_python_inc()

log(cmd)
if os.system(cmd) != 0:
  log('Failed to build extension')
  sys.exit(1)


extension_modules = []
setup(
    name="distnet",
    description="Fast convolution network library",
    long_description='',
    author="Russell Power & Justin Lin & Yury Skobov",
    author_email="justin.lin@nyu.edu",
    license="GPL",
    version="0.1",
    url="http://github.com/allenbo/fastnet",
    packages=[ 'distnet', 'cudaconv', 'garray'],
    package_dir={ 
      'distnet' : 'distnet',
      'cudaconv' : 'cudaconv' ,
      'garray' : 'garray',
    },
    requires=[
      'pycuda', 
      'numpy',
      'scikits.cuda',
    ],
    ext_modules = extension_modules)
