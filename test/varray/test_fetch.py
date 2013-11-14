import varray as va
import numpy as np

a = va.random.square_randn((3, 100, 100, 3) , slice_dim = (1, 2))
print a.local_shape

area = va.Area(va.Point(0, 45, 45, 0), va.Point(2,99, 99, 2))

rst = a.fetch(area)
print rst.shape

a = va.random.randn((4,4))
print a.local_shape
print a.local_data

a.gather()
print a.local_shape
print a.local_data
