#!/usr/bin/python2.7
'''
This test is for naive trainer to traine a full imagenet model
'''

from distnet import data, trainer, net, parser
import os
from distbase import util
import numpy as np

test_id = 'autostop_layerwise'

data_dir = '/ssd/nn-data/imagenet/'
checkpoint_dir = '/home/justin/fastnet/fastnet/checkpoint/'
base_checkpoint_path = '/hdfs/justin/checkpoint/imagenet_simple_base-30'
param_file = '/home/justin/fastnet/config/imagenet.cfg'
output_dir = ''
output_method = 'disk'

train_range = range(101, 1301) #1,2,3,....,40
test_range = range(1, 101) #41, 42, ..., 48
data_provider = 'imagenet'


train_dp = data.get_by_name(data_provider)(data_dir,train_range)
test_dp = data.get_by_name(data_provider)(data_dir, test_range)
checkpoint_dumper = trainer.CheckpointDumper(checkpoint_dir, test_id)
base_checkpoint_dumper = trainer.CheckpointDumper(base_checkpoint_path,"")

model = checkpoint_dumper.get_checkpoint()
if model is None:
  model = parser.parse_config_file(param_file)

save_freq = 1000
test_freq = 100
adjust_freq = 100
factor = 1
num_epoch = 15
learning_rate = 0.1
batch_size = 128
image_color = 3
image_size = 224
image_shape = (image_color, image_size, image_size, batch_size)
net = parser.load_model(net.FastNet(image_shape), model)


param_dict = globals()


class AutoStopLayerwisedTrainer(trainer.Trainer):
  def _finish_init(self):
    self.min_batch = self.save_freq * 10
    self.max_batch = self.save_freq * 100
    self.first_layer_name = self.net.get_first_active_layer_name()
    self.diff_log = os.path.join(self.checkpoint_dumper.checkpoint_dir, 'diff_log' + self.first_layer_name)
    self.diff_list = []
    self.threshold = 1e-6
    self.base_layers = self.base_checkpoint_dumper.get_checkpoint()['layers']

  def should_continue_training(self):
    if self.curr_batch < self.min_batch:
      return True
    elif self.diff_list[-1] < self.threshold * self.base_weight.size or self.diff_list[-1] >= np.mean(np.array(self.diff_list[-11:-1])) or self.curr_batch > self.max_batch:
      self.net.get_layer_by_name(self.first_layer_name).disable_bprop = True
      return False
    else:
      return True

  def _get_layer_weight(self, name):
    for layer in self.base_layers:
      if layer['name'] == name:
        return layer['weight'] + layer['bias'].transpose()

  def _log(self):
    with open(self.diff_log, 'a') as f:
      print >>f, self.net.get_first_active_layer_name()
      print >>f, self.diff_list[-1]

  def save_checkpoint(self):
    weight = self.net.get_weight_by_name(self.first_layer_name)
    self.base_weight = self._get_layer_weight(self.first_layer_name)
    weight_diff = weight - self.base_weight
    diff = np.sum(np.abs(weight_diff))
    self.diff_list.append(diff)
    util.log('%f', diff)
    self._log()

    trainer.Trainer.save_checkpoint(self)


for i in range(5):
  t = AutoStopLayerwisedTrainer(**param_dict)
  t.train(num_epoch)
