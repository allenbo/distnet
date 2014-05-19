import numpy as np
import math

cdef class Point(object):
  cdef public list point

  def __init__(self, first, *args):
    l = [first]
    for a in args:
      l.append(a)
    self.point = l

  def __len__(Point self): return len(self.point)

  def __str__(Point self):
    return str(self.point)

  def __reduce__(Point self):
    return (Point, tuple(self.point))

  def __richcmp__(Point self, Point other, int op):
    if op != 2 and op != 4: # 2 is equare and 4 is greater than
      raise Exception

    if op == 2:
      for i in xrange(len(self.point)):
        if self.point[i] != other.point[i]: return False
    else:
      for i in xrange(len(self.point)):
        if self.point[i] < other.point[i]: return False
    return True

  def __getitem__(self, i):
    return self.point[i]

  def __setitem__(self, i, t):
    self.point[i] = t

  def expand(self, padding):
    assert len(padding) == len(self.point)
    return Point(*[a + b for a, b in zip(self.point, padding)])


cdef class Area(object):
  cdef public Point _from, _to
  cdef public int dim

  def __init__(self, Point f, Point t):
    assert len(f) == len(t)

    self.dim = len(f)
    self._from = f
    self._to = t

  def __add__(self, other):
    _from = Point(*[ min(a, b) for a, b in zip(self._from.point, other._from.point)])
    _to = Point(* [ max(a, b) for a, b in zip(self._to.point, other._to.point)])
    return  Area(_from, _to)

  def __reduce__(Area self):
    return (Area, (self._from, self._to))

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

  def __repr__(self):
    return str(self)

  def __str__(self):
    return str(self._from) + ' to ' + str(self._to)

  def cmp(self, other):
    return self._from == other._from and self._to == other._to

  @property
  def id(self):
    return (tuple(self._from), tuple(self._to))

#  def __eq__(self, other):
#    return self._from == other._from and self._to == other._to
#
  def offset(self, point):
    ''' Key the shape of area, but move "point" offset, used with slice to find the slice tuple of two area, check ndarray '''
    _from = Point(*[a - b for a, b in zip(self._from.point, point.point)])
    _to = Point(*[ a - b for a, b in zip(self._to.point , point.point)])

    return Area(_from, _to)

  def move(self, point):
    ''' Key the shape of area, but move to "point" '''
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
