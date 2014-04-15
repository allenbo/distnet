from distbase.cuda_base import row_max_reduce, col_max_reduce
from distbase.cuda_base import find_row_max_id, find_col_max_id
from distbase.cuda_base import add_vec_to_rows, add_vec_to_cols
from distbase.cuda_base import div_vec_to_rows, div_vec_to_cols
from distbase.cuda_base import add_row_sum_to_vec, add_col_sum_to_vec
from distbase.cuda_base import same_reduce, same_reduce_multiview

from pycuda import gpuarray, autoinit
import numpy as np
import random


shapes = [(10, 10), (512, 10), (10, 512), (678231, 30), (30, 678231), (1026, 1026), (2049, 1996)]
for fd, sd in shapes:
  print 'build matrix'
  matrix_local = np.random.randn(fd, sd).astype(np.float32)
  max_local = np.zeros((1, fd)).astype(np.float32)
  maxid_local = np.zeros((1, fd)).astype(np.float32)
  
  matrix = gpuarray.to_gpu(matrix_local)
  max = gpuarray.to_gpu(max_local)
  maxid = gpuarray.to_gpu(maxid_local)
  
  print 'matrix.shape', matrix.shape

  row_max_reduce(max, matrix)
  max_local = matrix_local.max(axis = 1).reshape(max_local.shape)

  diff = np.abs(max_local - max.get())
  assert (diff < 1e-4).all()

  find_row_max_id(maxid, matrix)
  maxid_local = matrix_local.argmax(axis = 1).reshape(maxid_local.shape)

  diff = np.abs(maxid_local - maxid.get())
  assert (diff < 1e-4).all()

  max_local = np.zeros((1, sd)).astype(np.float32)
  maxid_local = np.zeros((1, sd)).astype(np.float32)
  max = gpuarray.to_gpu(max_local)
  maxid = gpuarray.to_gpu(maxid_local)

  col_max_reduce(max, matrix)
  max_local = matrix_local.max(axis = 0).reshape(max_local.shape)

  diff = np.abs(max_local - max.get())
  assert (diff < 1e-4).all()

  find_col_max_id(maxid, matrix)
  maxid_local = matrix_local.argmax(axis = 0).reshape(maxid_local.shape)

  diff = np.abs(maxid_local - maxid.get())
  assert (diff < 1e-4).all()

print 'row_max_reduce, col_max_reduce, find_row_max_id, find_col_max_id pass the test'

for fd, sd in shapes:
  print 'build matrix/vector'

  matrix_local = np.random.randint(20, size = (fd, sd)).astype(np.float32)
  result_local = np.zeros((fd, sd)).astype(np.float32)
  vec_local = np.random.randint(10, size = (fd, 1)).astype(np.float32)

  matrix = gpuarray.to_gpu(matrix_local)
  result = gpuarray.to_gpu(result_local)
  vec = gpuarray.to_gpu(vec_local)

  print 'matrix.shape', matrix.shape
  
  alpha = random.random()
  beta = random.random()
  add_vec_to_rows(matrix, vec, dest = result, alpha = alpha, beta = beta)
  result_local = matrix_local * beta + vec_local * alpha

  diff = np.abs(result_local - result.get())
  assert (diff < 1e-4).all()

  add_row_sum_to_vec(vec, matrix, alpha, beta)
  vec_local = (matrix_local.sum(1) * beta + vec_local.reshape((1, fd)) * alpha).reshape((fd, 1))
  
  diff = np.abs(vec_local - vec.get())
  assert (diff < 1e-4).all()

  div_vec_to_rows(matrix, vec, result)
  result_local = matrix_local / vec_local

  diff = np.abs(result_local - result.get())
  assert (diff < 1e-4).all()

  vec_local = np.random.randint(10, size = (1, sd)).astype(np.float32)
  vec = gpuarray.to_gpu(vec_local)

  add_vec_to_cols(matrix, vec, dest = result, alpha = alpha, beta = beta)
  result_local = matrix_local * beta + vec_local * alpha

  diff = np.abs(result_local - result.get())
  assert (diff < 1e-4).all()

  add_col_sum_to_vec(vec, matrix, alpha, beta)
  vec_local = (matrix_local.sum(0) * beta + vec_local * alpha)
  
  diff = np.abs(vec_local - vec.get())
  assert (diff < 1e-4).all()

  div_vec_to_cols(matrix, vec, result)
  result_local = matrix_local / vec_local

  diff = np.abs(result_local - result.get())
  assert (diff < 1e-4).all()

print 'add_vec_to_rows, add_vec_to_cols, div_vec_to_rows, div_vec_to_cols, add_row_sum_tovec, add_col_sum_to_vec pass the test'

sizes = [5, 10, 30, 128, 512, 2000, 10000, 789280]
for size in sizes:
  print 'build first/second'
  num_view = random.randint(1, 12)
  if num_view % 2 == 0:
    num_view += 1
  first_local = np.random.randint(2, size = (size, 1)).astype(np.float32)
  first_multi_local = np.random.randint(2, size = (size * num_view, 1)).astype(np.float32)
  second_local = np.random.randint(2, size = (size, 1)).astype(np.float32)
  
  first = gpuarray.to_gpu(first_local)
  first_multi = gpuarray.to_gpu(first_multi_local)
  second = gpuarray.to_gpu(second_local)

  print 'shape', first.shape

  count = same_reduce(second, first)
  assert (first_local == second_local).sum() == count

  count = same_reduce_multiview(second, first_multi, num_view)
  tmp = first_multi_local.reshape((num_view, size))
  u, indices = np.unique(tmp, return_inverse=True)
  max_occu = u[np.argmax(np.apply_along_axis(np.bincount, 0, indices.reshape(tmp.shape), None, np.max(indices) + 1), axis = 0)]
  max_occu = max_occu.reshape((size, 1))
  assert (max_occu == second_local).sum() == count
