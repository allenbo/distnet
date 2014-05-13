import numpy as np

class ConvDataLayout(object):
  CHANNEL = 0
  HEIGHT = 1
  WIDTH = 2
  BATCH = 3
  DIM = 4
  

class FilterLayout(object):
  CHANNEL = 0
  HEIGHT = 1
  WIDTH = 2
  NUM = 3
  DIM = 4

class FCDataLayout(object):
  NEURON = 0
  BATCH = 1
  DIM = 2

class WeightLayout(object):
  OUTPUT = 0
  INPUT = 1
  DIM = 2

backend = 'cudaconv'

def get_image_shape(color, image_y, image_x, batch_size):
  return (color, image_y, image_x, batch_size)

def get_filter_shape(channel, filter_size, num_filter):
  return (channel, filter_size, filter_size, num_filter)

def fold_image(input_shape):
  return (np.prod(input_shape[:-1]), input_shape[-1])
