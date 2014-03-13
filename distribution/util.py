import math

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def isinteger(value):
  try:
    int(value)
    return True
  except ValueError:
    return False

def divup(x, base):
  if x / base * base == x:
    return int(x / base)
  else:
    return int(x / base + 1)

def issquare(x):
  a = math.sqrt(x) 
  b = int(a)
  return a == b

def make_square_shape(rank, global_shape, num_worker):
  first , second = 1, 2
  nprow = math.sqrt(num_worker)
  assert first < second < len(global_shape), 'Wrong slice_dim ' + str(len(global_shape))
  local_nrow = global_shape[first] / nprow
  local_ncol = local_nrow

  first_pos = int(rank / nprow)
  second_pos = int(rank % nprow)

  first_from  = first_pos * local_nrow
  first_to = (first_pos + 1) * local_nrow  if num_worker - rank >= nprow else global_shape[first]
  second_from = second_pos * local_ncol
  second_to = (second_pos + 1) * local_ncol if (rank + 1) % nprow != 0  else global_shape[second]
  
  rst_shape = list(global_shape)
  rst_shape[first] = first_to - first_from
  rst_shape[second] = second_to - second_from
  return tuple(rst)

def make_stripe_shape(rank, global_shape, num_worker):
  slice_dim = 1
  nrow = global_shape[slice_dim] / num_worker

  pos_from = nrow * rank
  pos_to = (rank+ 1)* nrow
  if rank == num_workerk -1 :
    pos_to = global_shape[slice_dim]
  
  rst_shape = list(global_shape)
  rst_shape[slice_dim] = pos_to - pos_from
  return tuple(rst)
