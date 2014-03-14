import numpy as np
import math
from util import issquare

class Point(object):
  def __init__(self, first, *args):
    l = [first]
    for a in args:
      l.append(a)
    self.point = l

  def __len__(self): return len(self.point)

  def __str__(self):
    return str(self.point)

  def __eq__(self, other):
    return all([a == b for a, b in zip(self.point, other.point)])

  def __getitem__(self, i):
    return self.point[i]

  def __le__(self, other):
    return all([a <= b for a, b in zip(self.point, other.point)])

  def __gt__(self, other):
    return all([a >= b for a, b in zip(self.point, other.point)])

  def __setitem__(self, i, t):
    self.point[i] = t

  def expand(self, padding):
    assert len(padding) == len(self.point)
    return Point(*[a + b for a, b in zip(self.point, padding)])


class Area(object):
  def __init__(self, f, t):
    if len(f) != len(t):
      return None
    else:
      self.dim = len(f)
      self._from = f
      self._to = t

  def __add__(self, other):
    _from = Point(*[ min(a, b) for a, b in zip(self._from.point, other._from.point)])
    _to = Point(* [ max(a, b) for a, b in zip(self._to.point, other._to.point)])
    return  Area(_from, _to)

  @staticmethod
  def make_area(shape):
    _from = Point( *([0] * len(shape)) )
    _to = Point( *[ a - 1 for a in shape] )
    return Area(_from, _to)

  @property
  def slice(self):
    return tuple([slice(a, b + 1) for a, b in zip(self._from.point, self._to.point)])

  def __contains__(self, other):
    return all([ a <= b for a, b in zip(self._from.point, other._from.point)]) and all([ a >= b for a, b in zip(self._to.point, other._to.point)])

  def __and__(self, other):
    _from =  Point(*[ max(a, b) for a, b in zip(self._from.point, other._from.point)])
    _to = Point(*[ min(a, b) for a, b in zip(self._to.point, other._to.point)])

    if all([ a <= b for a, b in zip(_from.point, _to.point)]):
      return Area(_from, _to)
    else:
      return None
  def __str__(self):
    return str(self._from) + ' to ' + str(self._to)

  @property
  def id(self):
    return (tuple(self._from), tuple(self._to))

  def __eq__(self, other):
    return self._from == other._from and self._to == other._to

  def offset(self, point):
    _from = Point(*[a - b for a, b in zip(self._from.point, point.point)])
    _to = Point(*[ a - b for a, b in zip(self._to.point , point.point)])

    return Area(_from, _to)

  def move(self, point):
    _size = [ a -b for a, b in zip(self._to.point, self._from.point )]
    _from = Point(*point)
    _to = Point(*[a + b for a , b in zip(point, _size)])
    return Area(_from, _to)

  @property
  def shape(self):
    return tuple([ a - b + 1 for a, b in zip(self._to.point, self._from.point)])
  
  @property
  def size(self):
    return np.prod(self.shape)

  @staticmethod
  def min_from(area_list):
    rst = area_list[0]._from
    for area in area_list[1:]:
      if rst > area._from:
        rst = area._from
    import copy
    return copy.deepcopy(rst)


class DistMethod(object):
  square = 'square'
  stripe = 'stripe'

class VArray(object):
  def __init__(self, input_shape, num_worker, rank):
    self.global_shape = input_shape
    self.num_worker = num_worker
    self.rank = rank

    if issquare(self.num_worker):
      self.slice_dim = (1, 2)
      self.slice_method = DistMethod.square
      self.nprow = math.sqrt(self.num_worker)
      self.local_area = self.make_square_area(self.rank)
    else:
      self.slice_dim = 1
      self.slice_method = DistMethod.stripe
      self.local_area = self.make_stripe_area(self.rank)
    self.local_shape = self.local_area.shape
    self.global_area = Area.make_area(self.global_shape)

  def make_square_area(self, rank):
    first , second = self.slice_dim
    assert first < second < len(self.global_shape), 'Wrong slice_dim ' + str(len(self.global_shape))
    local_nrow = self.global_shape[first] / self.nprow
    local_ncol = local_nrow

    first_pos = int(rank / self.nprow)
    second_pos = int(rank % self.nprow)

    first_from  = first_pos * local_nrow
    first_to = (first_pos + 1) * local_nrow  if self.num_worker - rank >= self.nprow else self.global_shape[first]
    second_from = second_pos * local_ncol
    second_to = (second_pos + 1) * local_ncol if (rank + 1) % self.nprow != 0  else self.global_shape[second]

    _from = [0] * len(self.global_shape)
    _to = list(self.global_shape)

    _from[first] = int(first_from)
    _from[second] = int(second_from)
    _to[first] = int(first_to)
    _to[second] = int(second_to)
    _to = [x - 1 for x in _to]
    return Area(Point(*_from), Point(*_to))

  def make_stripe_area(self, rank):
    assert self.slice_dim < len(self.global_shape), 'Wrong slice dim'
    nrow = self.global_shape[self.slice_dim] / self.num_worker

    pos_from = nrow * rank
    pos_to = (rank+ 1)* nrow
    if rank == self.num_worker-1 :
      pos_to = self.global_shape[self.slice_dim]

    _from = [0] * len(self.global_shape)
    _to = list(self.global_shape)
    _from[self.slice_dim] = pos_from
    _to[self.slice_dim] = pos_to
    _to = [x - 1 for x in _to]
    return Area(Point(*_from) , Point(*_to))

  def cross_communicate(self, stride, filter_size, padding = 0, num_output = None):
    ''' When cross communicate is being called, FastNet is distribued the image cross height and width'''
    assert padding <= 0, str(padding)
    #r, c = self.slice_dim
    #The dimension of height and width
    r, c =  1, 2
    half_filter_size = (filter_size - 1) /2
    if stride != 1:
      global_row_begin_centroid = global_col_begin_centroid = half_filter_size + padding

      row_begin_centroid = global_row_begin_centroid
      col_begin_centroid = global_col_begin_centroid

      while row_begin_centroid < self.local_area._from[r]: row_begin_centroid += stride
      while col_begin_centroid < self.local_area._from[c]: col_begin_centroid += stride

      row_end_centroid = row_begin_centroid
      col_end_centroid = col_begin_centroid

      while row_end_centroid < self.local_area._to[r]: row_end_centroid += stride
      if row_end_centroid != self.local_area._to[r]:
        row_end_centroid -= stride
      while col_end_centroid < self.local_area._to[c]: col_end_centroid += stride
      if col_end_centroid != self.local_area._to[c]:
        col_end_centroid -= stride

      if num_output is not None:
        num_row , num_col = num_output
        diff = num_row - ((row_end_centroid - row_begin_centroid) / stride  + 1)
        if diff != 0:
          # change the centriod, asssume there are 4 GPU
          if self.local_area._from[r] == 0:
            row_end_centroid += diff * stride
          else:
            row_begin_centroid +=  diff * stride
        diff = num_col - ((col_end_centroid - col_begin_centroid) / stride  + 1)
        if diff != 0:
          # change the centriod, asssume there are 4 GPU
          if self.local_area._from[c] == 0:
            col_end_centroid += diff * stride
          else:
            col_begin_centroid +=  diff * stride

      row_up = half_filter_size - (row_begin_centroid - self.local_area._from[r])
      row_down = half_filter_size - (self.local_area._to[r] - row_end_centroid)
      col_left = half_filter_size - (col_begin_centroid - self.local_area._from[c])
      col_right = half_filter_size - (self.local_area._to[c] - col_end_centroid)
    else:
      row_up = row_down = col_left = col_right = half_filter_size

    import copy
    cross_from = copy.deepcopy(self.local_area._from)
    cross_to = copy.deepcopy(self.local_area._to)
    #not most top
    if self.local_area._from[r] != 0:
      cross_from[r] -= row_up
    #not most left
    if self.local_area._from[c] != 0:
      cross_from[c] -= col_left
    #not most down
    if self.local_area._to[r] != self.global_area._to[r]:
      cross_to[r] += row_down
    #not most right
    if self.local_area._to[c] != self.global_area._to[c]:
      cross_to[c] += col_right


    self.tmp_local_area = Area(cross_from, cross_to)
    overlapping = (np.prod(self.tmp_local_area.shape) - np.prod(self.local_shape)) * 4
    rst_shape = self.pad(padding, self.tmp_local_area.shape)
    return rst_shape, overlapping

  def pad(self, padding, old_shape):
    assert padding <= 0
    padding = -padding
    row, col = 1, 2
    new_shape = list(old_shape)
    #most down
    if self.local_area._to[row] == self.global_area._to[row]:
      new_shape[row] += padding
    #most right
    if self.local_area._to[col] == self.global_area._to[col]:
      new_shape[col] += padding

    return tuple(new_shape)
