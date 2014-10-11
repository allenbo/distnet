#!/usr/bin/python2.7

import pyximport
pyximport.install()
from distnet import data, trainer, net, parser
from garray import ConvDataLayout
from mpi4py import MPI

test_id = 'cifar-test'

#data_dir = '/proj/FastNet/exp/TestCentOSCuda/cifar-10.old/'
checkpoint_dir = 'checkpoint/'
param_file = 'config/cifar-13pct_new.cfg'


data_provider = 'cifar10'
batch_size = 1024
data_dir = '/home/justinlin/nn-data/cifar-10.%d/' % (batch_size)
if batch_size == 1024:
  train_range = range(1, 41) #1,2,3,....,40
  test_range = range(41, 50) #41, 42, ..., 48
elif batch_size == 128:
  train_range = range(1, 321) #1,2,3,....,40
  test_range = range(321, 380) #41, 42, ..., 48
else:
  assert False

train_dp = data.get_by_name(data_provider)(data_dir,train_range, minibatch_size = batch_size)
test_dp = data.get_by_name(data_provider)(data_dir, test_range, minibatch_size = batch_size)
checkpoint_dumper = trainer.CheckpointDumper(checkpoint_dir, test_id)

init_model = checkpoint_dumper.get_checkpoint()
if init_model is None:
  init_model = parser.parse_config_file(param_file)

save_freq = 10000
test_freq = 10000
adjust_freq = 100
factor = 1
num_epochs = 30
image_color = 3
image_size = 32
image_shape = ConvDataLayout.get_output_shape(image_size, image_size, image_color, batch_size)

net = parser.load_model(net.FastNet(image_shape), init_model)

param_dict = globals()
t = trainer.Trainer(**param_dict)
t.train()
