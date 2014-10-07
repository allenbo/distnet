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


test_id = 'imagenet_dummy'

data_dir = '/home/justinlin/nn-data/part-imagenet-category/'
checkpoint_dir = './checkpoint/'
param_file = './config/imagenet_cudaconv.cfg'
output_dir = ''
output_method = 'disk'

train_range = range(101, 103) #1,2,3,....,40
test_range = range(1, 3) #41, 42, ..., 48
data_provider = 'dummy'

image_size=224
batch_size=512
multiview = False
train_dp = data.get_by_name(data_provider)(image_size, 1000, batch_size = 1024, minibatch_size = batch_size)
test_dp = data.get_by_name(data_provider)(image_size, 1000, batch_size = 1024, minibatch_size = batch_size)
checkpoint_dumper = trainer.CheckpointDumper(checkpoint_dir, test_id)

model = checkpoint_dumper.get_checkpoint()
if model is None:
  model = parser.parse_config_file(param_file)

save_freq = 100
test_freq = 100
adjust_freq = 100
factor = 1
num_epochs = 90
image_color = 3
image_shape = ConvDataLayout.get_output_shape(image_size, image_size, image_color, batch_size)
net = parser.load_model(net.FastNet(image_shape), model)

param_dict = globals()
t = trainer.Trainer(**param_dict)
t.train()
