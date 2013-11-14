import cPickle as pickle
import numpy as np
import sys

debug = False

image_size = 27
num_channel = 96
num_image = 128
single_channel_p = ( image_size * image_size * num_image)

def compute(idx):
  channel = idx / single_channel_p

  idx = idx  % single_channel_p
  idx_image = idx / num_image
  image_id = idx % num_image

  num_row = idx_image / image_size
  num_col = idx_image % image_size

  return channel, num_row, num_col, image_id


bconv = 'rnormundo'
print 'Comparing', bconv, 'operation'
single_filename = bconv+'-output-single'
multi_filename = bconv+'-output-multi'

with open(single_filename) as f1, open(multi_filename) as f2:
  output1 = pickle.load(f1)
  output2 = pickle.load(f2)


if debug:
  print (output1[0, 200, :, 0] - output2[0, 200, :, 0]) / output1[0, 200, :, 0]

diff = abs(output1 - output2) / output1

index = diff.flatten().argsort()[-100:]

for idx in index:
  c, r, co, i =  compute(idx)
  print c, r, co, i, " " , diff[c, r, co, i], output1[c, r, co, i], output2[c, r, co, i]
