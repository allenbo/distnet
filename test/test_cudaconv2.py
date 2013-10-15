from pycuda import gpuarray, driver
from scipy.signal import convolve2d
import cudaconv2
import cudaconv2
import numpy as np
import pycuda.driver as cuda
import sys
cudaconv2.init()



def test_convolution():
  imgSize = 32
  filterSize = 5
  padding = 2
  color = 1
  imgNum = 1
  filterNum = 64
  
  stride = 1
  modulesX = 1 + int(((2 * padding + imgSize - filterSize) / float(stride)))
  
  print 'Modules X', modulesX
  
  
  img = gpuarray.to_gpu(np.ones((imgSize * imgSize * color, imgNum)).astype(np.float32))
  filter = gpuarray.to_gpu(np.ones((filterSize * filterSize * color, filterNum)).astype(np.float32))
  target = gpuarray.to_gpu(np.ones((modulesX * modulesX * filterNum, imgNum)).astype(np.float32))
  
  print 'standard output for convolution'
  print convolve2d(np.ones((imgSize, imgSize)).astype(np.float32), np.ones((filterSize, filterSize)).astype(np.float32),'valid')
  cudaconv2.convFilterActs(img, filter, target, imgSize, modulesX, modulesX, -padding, stride, color, 1, 0.0, 1.0)
  
  print 'pycuda output for convolution'
  atarget = target.get()
  
  print atarget