import numpy as np
import garray
import cProfile
from garray import transpose, sum
import time


shape = (128, 1020)
na = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)
a = garray.array(na)

b = transpose(a)

nb = np.transpose(na)
print b.get() - nb
print (b.get() - nb).sum()
start = time.time()
for i in range(100):
  b = transpose(a)

print 'time', time.time() -start
