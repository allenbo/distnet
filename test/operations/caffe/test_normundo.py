import caffe
from pycuda import gpuarray, driver
import numpy as np
from distbase.util import divup
caffe.init()


batch_size = 128
image_size = 27
color = 96
input_shape = (batch_size, color, image_size, image_size)

size = 5
scale = 0.0001
pow = 0.75
scaler = scale/(size * size)

output_size  = image_size
output_shape = input_shape

input_local = np.random.randn(*input_shape).astype(np.float32)
outgrad_local = np.random.randn(*input_shape).astype(np.float32)
output_local = np.random.randn(*output_shape).astype(np.float32)
ingrad_local = np.random.randn(*output_shape).astype(np.float32)
denom_local = np.random.randn(*output_shape).astype(np.float32)

input = gpuarray.to_gpu(input_local)
outgrad = gpuarray.to_gpu(outgrad_local)
output = gpuarray.to_gpu(output_local)
ingrad = gpuarray.to_gpu(ingrad_local)
denom = gpuarray.to_gpu(denom_local)

caffe.convResponseNormUndo(ingrad, denom, input, output, outgrad, color, size, image_size, scaler, pow, 0.0, 1.0)

scaler = scale / size
caffe.convResponseNormCrossMapUndo(ingrad, denom, input, output, outgrad, color, size,
    image_size, scaler, pow, False, 0.0, 1.0)
