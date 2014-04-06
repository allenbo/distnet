from pycuda import gpuarray, driver, autoinit
import numpy as np
import cudaconv
from distnet.util import divup


batch_size = 128
image_size = 55
color = 96
input_shape = (color, image_size, image_size, batch_size)

start = 0
pool_size = 3
stride = 2

output_size = divup(image_size - pool_size - start, stride) + 1
output_shape = (color, output_size, output_size, batch_size)

input_local = np.random.randn(*input_shape).astype(np.float32)
outgrad_local = np.random.randn(*input_shape).astype(np.float32)
output_local = np.random.randn(*output_shape).astype(np.float32)
ingrad_local = np.random.randn(*output_shape).astype(np.float32)

input = gpuarray.to_gpu(input_local)
outgrad = gpuarray.to_gpu(outgrad_local)
output = gpuarray.to_gpu(output_local)
ingrad = gpuarray.to_gpu(ingrad_local)

cudaconv.convLocalMaxUndo(input, ingrad, output, outgrad, pool_size, start, stride, output_size,
    output_size, image_size)
cudaconv.convLocalAvgUndo(ingrad, outgrad, pool_size, start, stride, output_size,
    output_size, image_size, image_size)
