from backend import cm_backend
import aux_operation
from aux_operation import sync_function, reshape_first, reshape_last
import numpy as np

convolution = sync_function(cm_backend.convFilterActs)

@sync_function
def bconvolution(*args):
  args = args[1:] + (1,)
  cm_backend.convImgActs(*args)

@sync_function
def wconvolution(*args):
  args = args + (1, 0)
  cm_backend.convWeightActs(*args)

# PoolLayer
maxpool = sync_function(cm_backend.convLocalMaxPool)
maxundo = sync_function(cm_backend.convLocalMaxUndo)
avgpool = sync_function(cm_backend.convLocalAvgPool)
@sync_function
def avgundo(*args):
  args = args[1:]
  cm_backend.convLocalAvgUndo(*args)

# RNormLayer
rnorm = sync_function(cm_backend.convResponseNorm)
@sync_function
def rnormundo(*args):
  args = args + (0.0, 1.0)
  cm_backend.convResponseNormUndo(*args)

rnormcrossmap = sync_function(cm_backend.convResponseNormCrossMap)
@sync_function
def rnormcrossmapundo(*args):
  args = args +  (0.0, 1.0)
  cm_backend.convResponseNormCrossMapUndo(*args)

def matrixmult(x, y, dest = None, alpha = 1.0, beta = 0.0):
  if len(x.shape) == 4:
    x = reshape_last(x)
  
  if len(y.shape) == 4:
    y = reshape_last(y)

  return aux_operation.matrixmult(x, y, dest, alpha, beta)

def transpose(mat, dest = None):
  if len(mat.shape) == 4:
    mat = reshape_last(mat)
  
  return aux_operation.transpose(mat, dest)

def convert_to_fc(input):
  if cm_backend.backend == 'cudaconv':
    return input
  
  if cm_backend.backend == 'caffe':
    batch_size = input.shape[cm_backend.ConvDataLayout.BATCH]
    new_shape = (batch_size, int(np.prod(input.shape) / batch_size))
    rst = transpose(input.reshape(new_shape))
    return rst

  assert False

def convert_to_conv(grad):
  if cm_backend.backend == 'cudaconv':
    return grad

  if cm_backend.backend == 'caffe':
    return transpose(grad)
