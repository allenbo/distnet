import varray as va
import numpy as np

a = va.zeros((100, 100))
print a.local_shape

b = va.random.square_randn((3, 100, 100, 3), slice_dim = (1, 2))
print b.local_shape

c = va.zeros_like(b)
print c.local_shape
