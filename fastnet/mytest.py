import cPickle
from pycuda import gpuarray, driver as cuda, autoinit
import numpy as np
import cudaconv2
from util import *

def load(filename):
  with open(filename, 'rb') as f:
    model = cPickle.load(f)
  return model


def print_matrix(x, name):
  print name
  a = x.get()[:, 0]
  for i in a:
    print '%.15f ' % i



'''
model = load('./checkpoint/test4-8.22')
net = FastNet(1.0,  (128, 3, 32, 32), 0, initModel = model)
'''

numColor = 1
numFilter = 64
filterSize = 5
imgSize = 32
outputSize = 32
padding = 2
stride = 1
batchSize = 1


'''
grad = gpuarray.to_gpu(np.ones((numFilter * outputSize * outputSize, batchSize), dtype = np.float32)
    * 2)
filter = gpuarray.to_gpu(np.ones((numColor * filterSize * filterSize, numFilter), dtype = np.float32))

outGrad = gpuarray.to_gpu(np.ones((numColor* outputSize * outputSize, batchSize), dtype = np.float32))

cudaconv2.convImgActs(grad, filter, outGrad, imgSize, imgSize, outputSize, -padding, stride, numColor, 1, 0.0, 1.0)
print_matrix(outGrad, 'grad')

input = gpuarray.to_gpu(np.ones((numColor * imgSize * imgSize, batchSize), dtype= np.float32) * 2)
weightGrad = gpuarray.to_gpu(np.ones((numColor * filterSize * filterSize, numFilter), dtype =
  np.float32))

cudaconv2.convWeightActs(input, grad, weightGrad, imgSize, outputSize, outputSize, filterSize, -padding, stride,
    numColor, 1, 0, 1, 1)

print_matrix(weightGrad, 'weight')
'''

offset = 1
a = gpuarray.to_gpu(np.random.randn(3, 4).astype(np.float32))
print a
b = gpuarray.GPUArray(shape=(3, 2), dtype = np.float32, gpudata = int(a.gpudata) + offset *a.dtype.itemsize, strides = a.strides)
c = transpose(b)
print c
print 

'''
gpu_partial_copy_to(a, b, 0, 3, 2, 5)
print a
print 
print b
'''
