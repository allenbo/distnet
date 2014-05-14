import pyximport
pyximport.install()

from varray.area import Area, Point
import numpy as np
import random
import copy

def make_shape(d):
  shape = []
  for i in range(d):
    shape.append(random.randint(10, 100))
  return shape

def make_point(n, f = None):
  point = []
  for i in range(n):
    if f is None:
      point.append(random.randint(10, 100))
    else:
      point.append(random.randint(f.point[i] + 1, 105))
  
  return Point(*point)

def make_padding(n):
  padding = []
  for i in range(n):
    padding.append(random.randint(2, 10))

  return padding

def test_point():
  for i in range(2, 5):
    p = make_point(i)
    assert len(p) == i, str(i) + str(len(p))
    
    # test ==, op = 2 for richcmp
    another_p = Point(*p.point)
    assert p == another_p

    another_p == make_point(i)
    greater = True
    for j in range(i):
      if p.point[j] < another_p.point[j]:
        greater = False
        break
    assert (p > another_p) == greater
    
    padding = make_padding(len(p))
    ep = p.expand(padding)
    
    assert (np.array(ep.point) == np.array(p) + np.array(padding)).all()
    
def test_area():
  for i in range(2, 5):
    # make_area
    shape = make_shape(i)    
    a = Area.make_area(shape)
    for j in range(i):
      assert a._from.point[j] == 0
      assert a._to.point[j] == shape[j] - 1
    
    # __add__
    f = make_point(i)
    t = make_point(i, f)
    
    a = Area(f, t)
    
    f2 = make_point(i)
    t2 = make_point(i, f2)

    a2 = Area(f2, t2)

    a3 = a2 + a;
    for j in range(i):
      assert a3._from.point[j] == min(f.point[j], f2.point[j])
      assert a3._to.point[j] == max(t.point[j], t2.point[j])

    #offset
    p = make_point(i)
    a2 = a.offset(p)
    
    for j in range(i):
      assert a2._from[j] == a._from[j] - p[j]
      assert a2._to[j] == a._to[j] - p[j]

    #move
    a2 = a.move(p)
    for j in range(i):
      assert a2._from[j] == p[j]
      assert a2._to[j] == a._to[j] - a._from[j] + p[j]

    #cmp
    b = copy.deepcopy(a)
    print 'a:', a
    print 'b:', b
    assert a.cmp(b)


if __name__ == '__main__':
  test_point()
  test_area()
