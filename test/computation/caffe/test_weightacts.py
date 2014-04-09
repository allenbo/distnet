import caffe
from pycuda import gpuarray, driver
import numpy as np
from distbase.util import divup
import time
caffe.init()

image_sizes = [224, 27, 13]
filter_sizes = [7, 5, 3]
colors = [3, 96, 256]
channels = [96, 256, 384]
paddings = [0,2,1 ]
strides = [2, 1, 1]

for image_size, color, channel, padding, stride, filter_size in zip(image_sizes, colors, channels, paddings, strides, filter_sizes):
  print 'color = %d channel = %d image_size = %d' % (color, channel, image_size)
  print '%10s\t%10s' %('batch','real')
  for batch_size in [32, 64, 128, 256]:
    input_shape = (batch_size, color, image_size, image_size)
    filter_shape = (channel, color, filter_size, filter_size)
    output_size = 1 + divup(2 * padding + image_size - filter_size, stride)
    ingrad_shape = (batch_size, channel, output_size, output_size)

    input_local = np.ndarray(input_shape).astype(np.float32)
    ingrad_local = np.ndarray(ingrad_shape).astype(np.float32)
    filter_local = np.zeros(filter_shape).astype(np.float32)

    input_caffe = gpuarray.to_gpu(input_local)
    ingrad_caffe = gpuarray.to_gpu(ingrad_local)
    filter_caffe = gpuarray.to_gpu(filter_local)

    caffe.convWeightActs(input_caffe, ingrad_caffe, filter_caffe, image_size, output_size,
        output_size, filter_size, -padding, stride, color, 1, 0)
    driver.Context.synchronize()


    count = 3
    start = time.time()
    for i in range(count):
      caffe.convWeightActs(input_caffe, ingrad_caffe, filter_caffe, image_size, output_size,
          output_size, filter_size, -padding, stride, color, 1, 0)
      driver.Context.synchronize()
    real_time = (time.time() - start) / count
    print '%10s\t%3.7f' %(batch_size, real_time)
