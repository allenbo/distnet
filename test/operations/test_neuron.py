from pycuda import gpuarray, driver
import numpy as np
import garray
garray.device_init()


batch_size = 128
image_size = 55
color = 3
input_shape = (color * image_size * image_size, batch_size)
output_shape = input_shape

e = 0.0

input_local = np.random.randn(*input_shape).astype(np.float32)
output_local = np.random.randn(*output_shape).astype(np.float32)

input = gpuarray.to_gpu(input_local)
output = gpuarray.to_gpu(output_local)

garray.relu_activate(input, output, e)

output_local = input_local
output_local[output_local < e] = 0

diff = output.get()- output_local
assert (diff < 1e-3).all()
print 'ReluNeuron passed the test'


a = 0.5
b = 0.5

garray.tanh_activate(input, output, a, b)

output_local = np.exp(output_local * (-2 * b)) + 1
output_local = a * (2.0 / output_local - 1)

diff = output.get()- output_local
assert (diff < 1e-3).all()
print 'TanhNeuron passed the test'
