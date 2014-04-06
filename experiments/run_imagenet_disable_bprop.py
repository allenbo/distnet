#!/usr/bin/python2.7
'''
This test is for naive trainer to traine a full imagenet model
'''

from distnet import data, trainer, net, parser

import sys
num_epochs = int(sys.argv[1])

test_id = 'layerwise-%d' % num_epochs

data_dir = '/ssd/nn-data/imagenet/'
checkpoint_dir = '/big0/checkpoints/'
param_file = './config/imagenet.cfg'
output_dir = ''
output_method = 'disk'

train_range = range(101, 1301) #1,2,3,....,40
test_range = range(1, 101) #41, 42, ..., 48
data_provider = 'imagenet'


train_dp = data.get_by_name(data_provider)(data_dir,train_range)
test_dp = data.get_by_name(data_provider)(data_dir, test_range)
checkpoint_dumper = trainer.CheckpointDumper(checkpoint_dir, test_id)

model = checkpoint_dumper.get_checkpoint()
if model is None:
  model = parser.parse_config_file(param_file)
  
save_freq = 100000
test_freq = 1000
adjust_freq = 100000
factor = 1
num_batch = 1
learning_rate = 0.1
batch_size = 128
image_color = 3
image_size = 224
image_shape = (image_color, image_size, image_size, batch_size)
net = parser.load_model(net.FastNet(image_shape), model)


param_dict = globals()

#num_batch = 1
#t = trainer.MiniBatchTrainer(**param_dict)
t = trainer.Trainer(**param_dict)

for layer in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']:
  t.train(num_epochs)
  net.get_layer_by_name(layer).disable_bprop()
