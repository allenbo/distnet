# Get backend
import os
backend_name = os.environ.get('BACKEND', 'cudaconv3')

if backend_name == 'cudaconv':
  import cudaconv as cm_backend
  from cudaconv import ConvDataLayout, FilterLayout, FCDataLayout, WeightLayout, backend_name
elif backend_name == 'cudaconv3':
  import cudaconv3 as cm_backend
  from cudaconv3 import ConvDataLayout, FilterLayout, FCDataLayout, WeightLayout, backend_name
else:
  import caffe as cm_backend
  from caffe import ConvDataLayout, FilterLayout, FCDataLayout, WeightLayout, backend_name

make_buffer = cm_backend.make_buffer
device_init = cm_backend.init
