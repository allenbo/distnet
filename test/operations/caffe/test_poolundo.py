import caffe
from pycuda import gpuarray, driver
import numpy as np
from distbase.util import divup
caffe.init()


batch_size = 128
image_size = 55
color = 96
input_shape = (batch_size, color, image_size, image_size)

start = 0
pool_size = 3
stride = 2

output_size = divup(image_size - pool_size - start, stride) + 1
output_shape = (batch_size, color, output_size, output_size)

input_local = np.random.randn(*input_shape).astype(np.float32)
outgrad_local = np.random.randn(*input_shape).astype(np.float32)
output_local = np.random.randn(*output_shape).astype(np.float32)
ingrad_local = np.random.randn(*output_shape).astype(np.float32)

input = gpuarray.to_gpu(input_local)
outgrad = gpuarray.to_gpu(outgrad_local)
output = gpuarray.to_gpu(output_local)
ingrad = gpuarray.to_gpu(ingrad_local)

print 'input.shape', input.shape, 'input.ptr', hex(input.ptr)
print 'output.shape', output.shape, 'output.ptr', hex(output.ptr)
print 'outgrad.shape', outgrad.shape,'outgrad.ptr', hex(outgrad.ptr)
print 'ingrad.shape', ingrad.shape, 'ingrad.ptr', hex(ingrad.ptr)

caffe.convLocalMaxUndo(input, ingrad, output, outgrad, pool_size, start, stride, output_size,
    output_size, image_size)
driver.Context.synchronize()
print 'max undo pass the test'
#caffe.convLocalAvgUndo(ingrad, outgrad, pool_size, start, stride, output_size,
#    output_size, image_size, image_size)
#driver.Context.synchronize()
#print 'avg undo pass the test'
