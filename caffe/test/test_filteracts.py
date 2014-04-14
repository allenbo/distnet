import caffe
from pycuda import gpuarray, driver
import numpy as np
from distbase.util import divup
caffe.init()

BATCH = 128
batch_size = BATCH


colors = [3, 96, 128]
channels = [96, 128, 128]
image_sizes = [224, 27, 13]
filter_sizes = [11, 5, 3]
paddings = [0, 2, 1]
strides = [4, 1, 1]

for image_size, color, channel, padding, stride, filter_size in zip(image_sizes, colors, channels, paddings, strides, filter_sizes):
  print 'color = %d channel = %d image_size = %d padding = %d stride = %d' % (color, channel, image_size, padding, stride)
  output_size = 1 + divup(2 * padding + image_size - filter_size, stride)

  input_shape = (batch_size, color, image_size, image_size)
  filter_shape = (channel, color, filter_size, filter_size)
  output_shape = (batch_size, channel, output_size, output_size)
  bias_shape = (channel, 1)

  print 'build input/output/filter/bias'
  # build input
  input_local = np.random.randint(10, size= input_shape).astype(np.float32)
  filter_local = np.random.randint(10, size = filter_shape).astype(np.float32)
  output_local = np.zeros(output_shape).astype(np.float32)

  # build bias
  bias_local = np.arange(np.prod(bias_shape)).astype(np.float32)
  np.random.shuffle(bias_local)
  bias_local = bias_local.reshape(bias_shape)

  input_caffe = gpuarray.to_gpu(input_local)
  filter_caffe = gpuarray.to_gpu(filter_local)
  output_caffe = gpuarray.to_gpu(output_local)
  bias_caffe = gpuarray.to_gpu(bias_local)
  
  print 'input.shape', input_caffe.shape
  print 'output.shape', output_caffe.shape
  print 'bias.shape', bias_caffe.shape
  print 'finished'

  print 'gpu computation'
  caffe.convFilterActs(input_caffe, filter_caffe, output_caffe, bias_caffe, image_size, output_size, output_size, -padding,
      stride, color, 1)
  print 'finished'

  print 'cpu computation'
  if padding != 0:
    tmp_shape  = (batch_size, color, image_size + 2 * padding, image_size + 2 * padding)
    tmp_input = np.zeros(tmp_shape).astype(np.float32)
    tmp_input[:, :, padding:padding+image_size, padding:padding+image_size] = input_local
    input_local = tmp_input


  batch_size = 1
  for b in range(batch_size):
    for c in range(channel):
      for x in range(output_size):
        for y in range(output_size):
          start_x = stride * x
          start_y = stride * y
          o = 0
          f = filter_local
          if start_x + filter_size >= input_local.shape[2]:
            f = f[:, :, :input_local.shape[2]-start_x, :]
          if start_y + filter_size >= input_local.shape[3]:
            f = f[:, :, :, :input_local.shape[3]-start_y]
          for cr in range(color):
            left = input_local[b, cr, start_x:start_x+filter_size, start_y:start_y+filter_size]
            right = f[c, cr, :, :]
            o += (left * right).sum()
          output_local[b, c, x, y] = o + bias_local[c, 0]
  
  print 'finished'

  output_caffe = output_caffe.get()
  diff = output_caffe[0, :, :, :] - output_local[0, :, :, :]
  assert (diff < 1e-3).all()
  print 'Convolution passed the test'
  batch_size = BATCH
