# Get backend
backend_name = 'cudaconv'

if backend_name == 'cudaconv':
  import cudaconv as cm_backend
  from cudaconv import ConvDataLayout, FilterLayout, FCDataLayout, WeightLayout
else:
  import caffe as cm_backend
  from caffe import ConvDataLayout, FilterLayout, FCDataLayout, WeightLayout

make_buffer = cm_backend.make_buffer
device_init = cm_backend.init
