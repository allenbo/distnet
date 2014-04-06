#!/usr/bin/python2.7
'''
This test is for naive trainer to traine a full imagenet model
'''

from distnet import data, trainer, net, parser
import pprint
from collections import deque
import os

test_id = 'imagenet_layerwised-2'

data_dir = '/ssd/nn-data/imagenet/'
checkpoint_dir = '/home/justin/fastnet/fastnet/checkpoint/'
param_file = '/home/justin/fastnet/config/imagenet.cfg'
output_dir = '/scratch1/justin/imagenet-pickle/'
output_method = 'disk'

train_range = range(101, 1301) #1,2,3,....,40
test_range = range(1, 101) #41, 42, ..., 48
data_provider = 'imagenet'


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
frag_epoch = 1
replaynet_epoch = 1
learning_rate = 0.1
batch_size = 128
image_color = 3
image_size = 224
image_shape = (image_color, image_size, image_size, batch_size)

train_layer_output_dumper = None
test_layer_output_dumper = None
if output_method == 'disk':
  if output_dir != '':
    train_layer_output_path = os.path.join(output_dir, 'train_data.pickle')
    train_layer_output_dumper = trainer.DataDumper(train_layer_output_path)
    test_layer_output_path = os.path.join(output_dir, 'test_data.pickle')
    test_layer_output_dumper = trainer.DataDumper(test_layer_output_path)
elif output_method == 'memory':
  train_layer_output_dumper = trainer.MemoryDataHolder()
  test_layer_output_dumper = trainer.MemoryDataHolder()

mynet = net.FastNet(learning_rate, image_shape, None)

param_dict = globals()

class ImageNetLayerwisedTrainer(trainer.Trainer):
  def divide_layers_to_stack(self):
    self.fc_params = []
    self.conv_params = []
    conv = True
    for ld in self.init_model:
      if ld['type'] in ['conv', 'rnorm', 'pool', 'neuron'] and conv:
        #self.conv_params.append(ld)
        self.conv_params.append(ld)
      elif ld['type'] == 'fc' or (not conv and ld['type'] == 'neuron'):
        self.fc_params.append(ld)
        conv = False
      else:
        self.softmax_param = ld

  def initialize_model(self):
    self.curr_model.extend(self.conv_stack['conv1'])
    self.curr_model.extend(self.conv_stack['conv2'])
    self.curr_model.extend(self.conv_stack['conv3'])
    self.curr_model.extend(self.conv_stack['conv4'])
    self.curr_model.extend(self.conv_stack['conv5'])
    self.curr_model.extend(self.fc_tmp)

  def _finish_init(self):
    self.final_num_epoch = self.num_epoch
    self.curr_model = []
    self.divide_layers_to_stack()
    self.conv_stack = net.FastNet.split_conv_to_stack(self.conv_params)
    self.fc_stack = net.FastNet.split_fc_to_stack(self.fc_params)


    self.fc_tmp = [self.fc_stack['fc8'][0], self.softmax_param]
    del self.fc_stack['fc8']
    self.stack = self.fc_stack

    self.initialize_model()
    pprint.pprint(self.stack)

    self.num_epoch = self.frag_epoch
    net = parser.load_from_checkpoint(param_file, 
                                      checkpoint_dumper.get_checkpoint(), 
                                      image_shape)

    self.container = deque()

  def report(self):
    pass

  def should_continue_training(self):
    return self.curr_epoch <= self.num_epoch

  def init_replaynet_data_provider(self):
    if self.output_method == 'disk':
      dp = data.get_by_name('intermediate')
      count = self.train_layer_output_dumper.get_count()
      self.train_dp = dp(self.train_layer_output_path, range(0, count), 'fc')
      count = self.test_layer_output_dumper.get_count()
      self.test_dp = dp(self.test_layer_output_path, range(0, count), 'fc')
    elif self.output_method == 'memory':
      dp = data.get_by_name('memory')
      self.train_dp = dp(self.train_layer_output_dumper)
      self.test_dp = dp(self.test_layer_output_dumper)

  def train_replaynet(self, stack):
    self.container.append(self.save_freq)
    self.container.append(self.test_freq)
    self.container.append(self.train_dp)
    self.container.append(self.test_dp)
    self.container.append(self.train_layer_output_dumper)
    self.container.append(self.test_layer_output_dumper)
    self.container.append(self.net)

    self.save_freq = self.curr_batch + 100
    self.test_freq = self.curr_batch + 100
    print self.save_freq
    self.curr_batch = self.curr_epoch = 0
    self.init_replaynet_data_provider()

    model = []
    model.extend(stack)
    model.extend(self.fc_tmp)

    self.train_layer_output_dumper = None
    self.test_layer_output_dumper = None
    size = self.net['fc8'].get_input_size()
    image_shape = (size, 1, 1, self.batch_size)
    self.net = net.FastNet(self.learning_rate, image_shape, model)
    self.replaynet = self.net
    self.num_epoch = self.replaynet_epoch
    trainer.Trainer.train(self, self.replaynet_epoch)

    self.net = self.container.pop()
    self.test_layer_output_dumper = self.container.pop()
    self.test_layer_output_dumper.reset()
    self.train_layer_output_dumper = self.container.pop()
    self.train_layer_output_dumper.reset()
    self.test_dp = self.container.pop()
    self.train_dp = self.container.pop()
    self.test_freq = self.container.pop()
    self.save_freq = self.container.pop()


  def reset_trainer(self, i):
    if i == len(self.stack) - 1:
      self.num_epoch = self.final_num_epoch
    else:
      self.num_epoch = self.frag_epoch

    self.curr_batch = self.curr_epoch = 0
    self.init_data_provider()

  def train(self, num_epochs):
    trainer.Trainer.train(self, self.num_epoch)
    for i, stack in enumerate(self.stack.values()):
      self.train_replaynet(stack)
      self.reset_trainer(i)

      self.net.drop_layer_from('fc8')

      for layer in self.replaynet:
        if layer.type != 'data':
          self.net.append_layer(layer)
      trainer.Trainer.train(self, self.num_epoch)


t = ImageNetLayerwisedTrainer(**param_dict)
t.train(num_epoch)
