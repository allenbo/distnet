#!/usr/bin/python2.7
'''
This test is for naive trainer to traine a full imagenet model
'''

import pyximport
pyximport.install()
from distnet import data, trainer, net, parser
from mpi4py import MPI
from garray import ConvDataLayout
from distbase import util


test_id = 'imagenet-simple'

#checkpoint_dir = '/ssd/justin/checkpoints/'
checkpoint_dir = './checkpoints/'
param_file = './config/imagenet.cfg'
output_dir = ''
output_method = 'disk'

#data_dir = '/ssd/nn-data/imagenet/'
#train_range = range(101, 1301) #1,2,3,....,40
#test_range = range(1, 101) #41, 42, ..., 48
data_dir = '/ssd/justin/part-imagenet-category/'
train_range = range(11, 100) #1,2,3,....,40
test_range = range(1, 10) #41, 42, ..., 48
data_provider = 'imagenet'


image_size=224
batch_size=128
multiview = False
train_dp = data.get_by_name(data_provider)(data_dir,train_range, batch_size = 1024)
test_dp = data.get_by_name(data_provider)(data_dir, test_range, batch_size = 1024)
checkpoint_dumper = trainer.CheckpointDumper(checkpoint_dir, test_id)

model = checkpoint_dumper.get_checkpoint()
if model is None:
  model = parser.parse_config_file(param_file)

save_freq = 100
test_freq = 100
num_epoch = 30
image_color = 3
image_shape = ConvDataLayout.get_output_shape(image_size, image_size, image_color, batch_size)
net = parser.load_model(net.FastNet(image_shape), model)

param_dict = globals()
t = trainer.Trainer(**param_dict)

t.train(num_epoch)