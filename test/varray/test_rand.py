import varray
from varray import VArray, DistMethod
import numpy as np

a = varray.random.randn((1000, 20))
print a.local_shape

a = varray.random.square_randn((3, 200, 200, 3), slice_dim = (1,2))
print a.local_shape


a = np.random.randn(3, 200, 200, 3).astype(np.float32)
a = varray.square_array(a, slice_dim = (1, 2))
print a.local_shape

seed = varray.get_seed()
print seed
import random
random.seed(seed)
np.random.seed(seed)

a = range(10)
random.shuffle(a)
print a
a = np.arange(10)
np.random.shuffle(a)
print a
