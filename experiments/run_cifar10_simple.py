#!/usr/bin/python2.7

from fastnet import data, trainer, net, parser

test_id = 'cifar-test-1'

data_dir = '/ssd/nn-data/cifar-10.old/'
checkpoint_dir = './checkpoint/'
param_file = 'config/cifar-10-26pct.cfg'

train_range = range(1, 41) #1,2,3,....,40
test_range = range(41, 49) #41, 42, ..., 48
data_provider = 'cifar10'


train_dp = data.get_by_name(data_provider)(data_dir,train_range)
test_dp = data.get_by_name(data_provider)(data_dir, test_range)
checkpoint_dumper = trainer.CheckpointDumper(checkpoint_dir, test_id)

init_model = checkpoint_dumper.get_checkpoint()
if init_model is None:
  init_model = parser.parse_config_file(param_file)
  

save_freq = 100
test_freq = 100
adjust_freq = 100
factor = 1
num_epoch = 15
learning_rate = 1.0
batch_size = 128
image_color = 3
image_size = 32
image_shape = (image_color, image_size, image_size, batch_size)

net = parser.load_model(net.FastNet(image_shape), init_model)

param_dict = globals()
print type(param_dict)
t = trainer.Trainer(**param_dict)
t.train(num_epoch)
