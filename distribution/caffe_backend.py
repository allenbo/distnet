import numpy as np
class ConvDataLayout(object):
  BATCH = 0
  CHANNEL = 1
  HEIGHT = 2
  WIDTH = 3
  DIM = 4
  

class FilterLayout(object):
  NUM = 0
  CHANNEL = 1
  HEIGHT = 2
  WIDTH = 3
  DIM = 4

class FCDataLayout(object):
  NEURON = 0
  BATCH = 1
  DIM = 2

class WeightLayout(object):
  OUTPUT = 0
  INPUT = 1
  DIM = 2

backend = 'caffe'

def get_image_shape(color, image_y, image_x, batch_size):
  return (batch_size, color, image_y, image_x)

def get_filter_shape(channel, filter_size, num_filter):
  return (num_filter, channel, filter_size, filter_size)

def fold_image(input_shape):
  return (np.prod(input_shape[1:]), input_shape[0])
