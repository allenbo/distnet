import caffe
from pycuda import gpuarray, driver, autoinit
import numpy as np
from distnet.util import divup
import time
import test_base
np.set_printoptions(threshold = np.nan)

expectation_time = []
real_time = []
percent = []
data_amount = []
comput_amount = []

image_sizes = [224, 27, 13]
filter_sizes = [11, 5, 3]
colors = [3, 96, 256]
channels = [96, 256, 384]
paddings = [0, 0, 0]
strides = [4, 1, 1]

band_width = int(test_base.memory_bandwidth(0))

for image_size, color, channel, padding, stride, filter_size in zip(image_sizes, colors, channels, paddings, strides, filter_sizes):
  print 'color = %d channel = %d image_size = %d' % (color, channel, image_size)
  print '%10s\t%10s\t%10s\t%10s\t%10s\t%10s' %('batch', 'data', 'expect', 'real', 'percent', 'comput')
  for batch_size in [128, 256]: #[32, 64, 128, 256]:
    input_shape = (batch_size, color, image_size, image_size)
    filter_shape = (channel, color, filter_size, filter_size)
    output_size = 1 + divup(2 * padding + image_size - filter_size, stride)
    output_shape = (batch_size, channel, output_size, output_size)

    input_local = np.random.randn(*input_shape).astype(np.float32)
    filter_local = np.random.randn(*filter_shape).astype(np.float32)
    output_local = np.zeros(output_shape).astype(np.float32)

    input = gpuarray.to_gpu(input_local)
    filter = gpuarray.to_gpu(filter_local)
    output = gpuarray.to_gpu(output_local)

    count = 2
    for i in range(count):
      caffe.convFilterActs(input, filter, output, image_size, output_size, output_size, -padding, stride, color, 1)
      driver.Context.synchronize()

    count = 10
    start = time.time()
    for i in range(count):
      caffe.convFilterActs(input, filter, output, image_size, output_size, output_size, -padding, stride, color, 1)
      driver.Context.synchronize()
    real_time.append((time.time() - start) / count)

    blockpermodule = filter_shape[0] / 32
    data_amount.append(((np.prod(output_shape[1:-1]) * (np.prod(filter_shape) / blockpermodule +
      np.prod(filter_shape[:-1]) * batch_size) + np.prod(output_shape)) * 4.0)/(1<<20))
    comput_amount.append( (np.prod(output_shape) * np.prod(filter_shape[:-1])) / 1e9)
    expectation_time.append(data_amount[-1] *(1<<20) * 1.0 / (band_width * 1e9))
    percent.append(expectation_time[-1] / real_time[-1])
    print '%10s\t%3.7f\t%3.7f\t%3.7f\t%3.7f\t%4.6f' %(batch_size, data_amount[-1], expectation_time[-1],
        real_time[-1], percent[-1], comput_amount[-1])
