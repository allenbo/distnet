#!/usr/bin/env python

import cPickle, os
import numpy as N

BATCHSIZE = 1024

batches = [cPickle.load(open('./old/data_batch_%d' % d)) for d in range(1, 6)]

def tolist(a):
  if isinstance(a, list): return a
  assert len(a.shape) == 2
  a = a.tolist()
  return a[0]

merged_data = batches[0]['data']
merged_labels = tolist(batches[0]['labels'])

for b in batches[1:]:
  merged_data = N.concatenate((merged_data, b['data']), axis=1)
  merged_labels = merged_labels + tolist(b['labels'])

print 'There are %d images' % (len(merged_labels))
index = 0
i = 0
while index < len(merged_labels):
    batch = {}
    batch['data'] = merged_data[:, index:index+BATCHSIZE]
    batch['labels'] = merged_labels[index:index+BATCHSIZE]

    with open('./data_batch_%d' % (i + 1), 'w') as f:
      f.write(cPickle.dumps(batch, protocol=-1))
      f.close()
    
    i += 1
    index += BATCHSIZE
