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
cmd = 'cd cudaconv && make'

log(cmd)
if os.system(cmd) != 0:
  log('Failed to build extension')
  sys.exit(1)

log('About to build cudaconv3 extension.')
cmd = 'cd cudaconv3 && make'

log(cmd)
if os.system(cmd) != 0:
  log('Failed to build extension')
  sys.exit(1)

log('About to build caffe extension.')
cmd = 'cd caffe && make'

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
    url="http://github.com/allenbo/distnet",
    packages=[ 'distnet', 'cudaconv', 'cudaconv3', 'caffe', 'garray', 'varray', 'distribution', ],
    package_dir={
      'distnet' : 'distnet',
      'cudaconv' : 'cudaconv' ,
      'cudaconv3' : 'cudaconv3' ,
      'caffe' : 'caffe', 
      'garray' : 'garray',
      'varray' : 'varray',
      'distbase': 'distbase',
    },
    requires=[
      'pycuda', 
      'numpy',
      'scikits.cuda',
    ],
    ext_modules = extension_modules)
