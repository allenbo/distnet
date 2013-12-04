import numpy as np

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
    return rst
