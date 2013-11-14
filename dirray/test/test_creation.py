from varray.ndarray import VArray, DistMethod
import numpy as np

b = np.random.randn(3, 201, 201, 3).astype(np.float32)
a = VArray(b,  slice_method = DistMethod.Square, slice_dim = (1, 2))

print a.local_shape
