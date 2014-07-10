from backend import cm_backend
import aux_operation
from aux_operation import sync_function, reshape_first, reshape_last
import numpy as np
from new_gpuarray import sum, max, iexp, copy_to, to_gpu


convert_from_data = cm_backend.convert_from_data
convert_to_fc = cm_backend.convert_to_fc
convert_to_conv = cm_backend.convert_to_conv
convert_from_backend = cm_backend.convert_from_backend

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

def random_uniform(shape):
  size = np.prod(shape)
  obj = np.random.uniform(0, 1, size).astype(np.float32).reshape(shape)
  return to_gpu(obj)


def fcforward(input, output, weight, bias, prev_conv):
  if prev_conv:
    real_input = convert_to_fc(input)
  else:
    real_input = input

  matrixmult(weight, real_input, dest = output)
  copy_to(output + bias, output)


def fcbackward(input, weight, grad, out_grad, weight_grad, bias_grad, prev_conv):
  matrixmult(transpose(weight), grad, dest = out_grad)
  
  if prev_conv:
    copy_to(convert_to_conv(out_grad), out_grad)
    real_input = convert_to_fc(input)
  else:
    real_input = input

  matrixmult(grad, transpose(real_input), dest = weight_grad)
  copy_to(sum(grad, axis = 1), bias_grad)


def softmax(input, output):
  max_rst = max(input, axis = 0)
  copy_to(input - max_rst, output)
  iexp(output)
  sum_rst = sum(output, axis = 0)
  copy_to(output / sum_rst, output)
