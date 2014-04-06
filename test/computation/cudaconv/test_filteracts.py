import cudaconv
from pycuda import gpuarray, driver
import numpy as np
from distbase.util import divup
import time
import test_base
cudaconv.init()

expectation_time = []
real_time = []
percent = []
data_amount = []
comput_amount = []

image_sizes = [224, 27, 13]
filter_sizes = [11, 5, 3]
colors = [3, 96, 256]
channels = [96, 256, 384]
paddings = [0, 2, 1]
strides = [4, 1, 1]

band_width = int(test_base.memory_bandwidth(0))

for image_size, color, channel, padding, stride, filter_size in zip(image_sizes, colors, channels, paddings, strides, filter_sizes):
  print 'color = %d channel = %d image_size = %d' % (color, channel, image_size)
  print '%10s\t%10s' %('batch', 'real')
  for batch_size in [32, 64, 128, 256]:
    input_shape = (color, image_size, image_size, batch_size)
    filter_shape = (color, filter_size, filter_size, channel)
    output_size = 1 + divup(2 * padding + image_size - filter_size, stride)
    output_shape = (channel, output_size, output_size, batch_size)

    input_local = np.random.randn(*input_shape).astype(np.float32)
    filter_local = np.random.randn(*filter_shape).astype(np.float32)
    output_local = np.zeros(output_shape).astype(np.float32)

    input = gpuarray.to_gpu(input_local)
    filter = gpuarray.to_gpu(filter_local)
    output = gpuarray.to_gpu(output_local)

    cudaconv.convFilterActs(input, filter, output, image_size, output_size, output_size, -padding, stride, color, 1)
    driver.Context.synchronize()

    count = 3
    start = time.time()
    for i in range(count):
      cudaconv.convFilterActs(input, filter, output, image_size, output_size, output_size, -padding, stride, color, 1)
      driver.Context.synchronize()
    real_time.append((time.time() - start) / count)

    print '%10s\t%3.7f' %(batch_size, real_time[-1])
