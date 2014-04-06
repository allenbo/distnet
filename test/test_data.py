from distnet import data, util
from distbase.util import print_matrix
import numpy as np
import time

def test_imagenet_loader():
  #df = data.get_by_name('imagenet')(
  #                               '/ssd/nn-data/imagenet/', 
  #                               batch_range=range(1000))
  df = data.get_by_name('dummy')(224, 1000)
  
  for i in range(32):
    st = time.time()
    batch = df.get_next_batch(8 * 10)
    print time.time() - st
    #print batch.labels
    print batch.data.shape
    time.sleep(0.5)
  print batch.labels


def test_cifar_loader():
  data_dir = '/ssd/nn-data/cifar-10.old/'
  dp = data.get_by_name('cifar10')(data_dir, [1])
  batch_size = 128
  
  data_list = []
  for i in range(11000):
    batch = dp.get_next_batch(batch_size)
    batch = batch.data.get()
    data_list.append(batch)

    if batch.shape[1] != batch_size:
      break
  batch = np.concatenate(data_list, axis = 1)
  #print_matrix(batch, 'batch')
  
if __name__ == '__main__':
  test_imagenet_loader()
  test_cifar_loader()
