import util
from util import divup
from pycuda import driver, gpuarray
from pycuda.gpuarray import GPUArray
from pycuda.compiler import SourceModule
from pycuda.elementwise import ElementwiseKernel
from pycuda.gpuarray import GPUArray
import numpy as np

from scikits.cuda import cublas
sgemm = None

def _initialize_cublas():
  global sgemm
  if sgemm:
    return 
  try:
    cublas.cublasInit()
    sgemm = cublas.cublasSgemm
  except AttributeError:
    handle = cublas.cublasCreate()
    def sgemm(*args):
      cublas.cublasSgemm(handle, *args)

class CompiledSource(object):
  '''
  Compile a source string with PyCuda, caching the resulting module.
  '''
  def __init__(self, src, kernel_name):
    self.src = src
    self.kernel_name = kernel_name
    self.module = None
    self.kernel = None


  def __call__(self, *args, **kw):
    if self.module is None:
      util.log('Compiling... %s', self.kernel_name)
      self.module = SourceModule(self.src)
      self.kernel = self.module.get_function(self.kernel_name)

    self.kernel(*args, **kw)
    driver.Context.synchronize()

def I(i): return np.int32(i)
def F(f): return np.float32(f)


_transpose_diagonal_ = CompiledSource(
'''
__global__ void transposeDiagonal(float *idata,
 float *odata, int height,int width)
{
  const int TILE_DIM = 32;
  const int BLOCK_ROWS = 8;
  __shared__ float tile[TILE_DIM][TILE_DIM+1];
  int blockIdx_x, blockIdx_y;

  // diagonal reordering
  if (width == height) {
    blockIdx_y = blockIdx.x;
    blockIdx_x = (blockIdx.x+blockIdx.y)%gridDim.x;
  } else {
    int bid = blockIdx.x + gridDim.x*blockIdx.y;
    blockIdx_y = bid%gridDim.y;
    blockIdx_x = ((bid/gridDim.y)+blockIdx_y)%gridDim.x;
  }

  int xIndex = blockIdx_y*TILE_DIM + threadIdx.x;
  int yIndex = blockIdx_x*TILE_DIM + threadIdx.y;
  int index_out = xIndex + (yIndex)*height;

  xIndex = blockIdx_x*TILE_DIM + threadIdx.x;
  yIndex = blockIdx_y*TILE_DIM + threadIdx.y;
  int index_in = xIndex + (yIndex)*width;
  for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
    tile[threadIdx.y+i][threadIdx.x] = idata[index_in+i*width];
  }

  __syncthreads();

  for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
    odata[index_out+i*height] = tile[threadIdx.x][threadIdx.y+i];
  }
}
''', 'transposeDiagonal')

_transpose_coalesced_ = CompiledSource('''
__global__ 
void transposeCoalesced(const float *idata, float *odata,int rows, int cols)
{
  const int TILE_DIM = 32;
  const int BLOCK_ROWS = 8;
  __shared__ float tile[TILE_DIM][TILE_DIM];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  if(x >= cols) return;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
     if(y + j >= rows) break;
     tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*cols + x];
  }

  __syncthreads();

  //x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
  //y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    if(y + j >= rows) continue;
     odata[x*rows + y + j] = tile[threadIdx.y+j][threadIdx.x];
  }
}
    ''', 'transposeCoalesced')



@util.timed_fn
def transpose(mat, dst = None):
  mh, mw = mat.shape
  if dst is None:
    dst = gpuarray.empty((mw, mh), dtype=np.float32)

  block = (32, 8, 1)
  grid = (divup(mw, 32), divup(mh, 32))
  sleading = mat.strides[0] / 4
  dleading = dst.strides[0] / 4
  if mh % 32 == 0 and mw % 32 == 0:
    _transpose_diagonal_(mat, dst, I(mh), I(mw), block=block, grid=grid)
  else:
    _transpose_coalesced_(mat, dst, I(mh), I(mw), block=block, grid=grid)

  return dst
@util.lazyinit(_initialize_cublas)
@util.timed_fn
def matrixmult(x, y, dest = None, alpha = 1.0, beta = 0.0):
  if isinstance(x, GPUArray):
    m = y.shape[1]
    n = x.shape[0]
    k = x.shape[1]

    assert k == y.shape[0], (x.shape, y.shape)

    if dest is None:
      result = GPUArray((n, m), dtype=x.dtype)
    else:
      result = dest
    sgemm('n', 'n', m, n, k, alpha, y.gpudata, m, x.gpudata,
        k, beta, result.gpudata, m)

    driver.Context.synchronize()
    return result
  else:
    return np.dot(x, y)

