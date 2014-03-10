import garray
import numpy as np
import time

num = 10
a = np.random.randn(96, 64, 64, 128).astype(np.float32)
ga = garray.array(a)
print 'time to start testing'
def test_cpu():
  for i in range(num):
    tmp = ga.get()[:, 0:32, 0:32, :].copy()
    gb = garray.array(tmp)

def test_gpu():
  for i in range(num):
    gb = ga[:, 0:32, 0:32, :]



tests = [test_cpu, test_gpu]

for test in tests:
  start = time.time()
  test()
  print test.__name__, 'time =', time.time() -start
