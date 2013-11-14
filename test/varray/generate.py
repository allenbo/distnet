import numpy as np
import sys
import cPickle as pickle


assert len(sys.argv) == 6 or len(sys.argv) == 7

initW = 1.0

if len(sys.argv) == 7:
  initW = float(sys.argv[6])
  print initW
  del sys.argv[6]

filename = sys.argv[1]
shape = tuple([int(x) for x in sys.argv[2:]])
a = np.random.randn(*shape).astype(np.float32) * initW

with open(filename, 'w') as f:
  pickle.dump(a, f, protocol=-1)
