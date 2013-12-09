import cPickle as pickle
import numpy as np
import sys

debug = False

assert len(sys.argv) == 2

print 'Comparing', sys.argv[1], 'operation'
single_filename = sys.argv[1]+'-output-single'
multi_filename = sys.argv[1]+'-output-multi'

with open(single_filename) as f1, open(multi_filename) as f2:
  output1 = pickle.load(f1)
  output2 = pickle.load(f2)

print 'single version shape', output1.shape
print 'multi version shape', output2.shape

if debug:
  print (output1[0, 0, :, 0] - output2[0, 0, :, 0])

print output1[0, 0, :, 0]
print output2[0, 0, :, 0]
diff = abs(output1 - output2)
#print diff[0, 0, :, 0]

#print diff.flatten().argsort()[-100:]
assert (diff < 1e-5).all()

print 'They are the same'
