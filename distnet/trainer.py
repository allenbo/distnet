from distbase import util
from distbase.util import log, timer
from collections import deque
from distnet import argparse, layer, data
from distnet.checkpoint import CheckpointDumper, DataDumper, MemoryDataHolder
from distnet.layer import TRAIN, TEST
from distnet.net import FastNet
from distnet.parser import parse_config_file, load_model
from distnet.scheduler import Scheduler
import os
import sys
import time
import math
from garray import ConvDataLayout


def cache_outputs(net, dp, dumper, layer_name = 'pool5', index = -1):
  '''
  fprop ``net`` through an entire epoch, saving the output of ``layer_name`` into ``dumper``. 
  :param net:
  :param layer_name:
  :param dp:
  :param dumper:
  '''
  dp.reset()
  curr_batch = 0
  batch = dp.get_next_batch(128)
  epoch = batch.epoch
  
  if layer_name != '':
    index = net.get_output_index_by_name(layer_name)
  
  while epoch == batch.epoch:
    batch_start = time.time()
    net.train_batch(batch.data, batch.labels, TEST)
    cost, correct, numCase = net.get_batch_information()
    curr_batch += 1
    log('%d.%d: error: %f logreg: %f time: %f', epoch, curr_batch, 1 - correct, cost, time.time() - batch_start)
    dumper.add({ 'labels' : batch.labels.get(),
                 'fc' : net.get_output_by_index(index).get().transpose()})
    batch = dp.get_next_batch(128)
  dumper.flush()


# Trainer should take: (training dp, test dp, distnet, checkpoint dir)
class Trainer:
  def __init__(self, checkpoint_dumper, train_dp, test_dp, batch_size, net=None, **kw):
    self.checkpoint_dumper = checkpoint_dumper
    self.train_dp = train_dp
    self.test_dp = test_dp
    self.batch_size = batch_size
    self.net = net
    self.curr_batch = self.curr_epoch = 0
    self.annealing_factor = 10
    self.multiview = False
    self.start_time = time.time()

    for k, v in kw.iteritems():
      setattr(self, k, v)


    checkpoint = self.checkpoint_dumper.get_checkpoint()
    if checkpoint:
      self.train_outputs = checkpoint['train_outputs']
      self.test_outputs = checkpoint['test_outputs']
      self.base_time = self.train_outputs[-1][-1]
    else:
      self.train_outputs = []
      self.test_outputs = []
      self.base_time = 0

    self._finish_init()

  def _finish_init(self):
    pass

  def init_data_provider(self):
    self.train_dp.reset()
    self.test_dp.reset()


  def annealing(self):
    self.net.adjust_learning_rate( 1.0 / self.annealing_factor)
    self.net.print_learning_rates()

  def save_checkpoint(self):
    model = {}
    model['layers'] = self.net.get_dumped_layers()
    model['train_outputs'] = self.train_outputs
    model['test_outputs'] = self.test_outputs

    log('---- save checkpoint ----')
    self.print_net_summary()
    self.checkpoint_dumper.dump(checkpoint=model, suffix=self.curr_epoch)


  def adjust_lr(self):
    log('---- adjust learning rate ----')
    self.net.adjust_learning_rate(self.factor)

  def elapsed(self):
    return time.time() - self.start_time + self.base_time

  def get_test_error(self):
    batch_size = self.batch_size
    test_data = self.test_dp.get_next_batch(batch_size)

    input, label = test_data.data, test_data.labels
    if self.multiview:
      num_view = self.test_dp.num_view
      self.net.test_batch_multiview(input, label, num_view)
      cost , correct, numCase = self.net.get_batch_information_multiview(num_view)
    else:
      self.net.train_batch(input, label, TEST)
      cost , correct, numCase, = self.net.get_batch_information()

    self.test_outputs += [({'logprob': [cost, 1 - correct]}, numCase, self.elapsed())]
    log( '---- test ----')
    log('error: %f logreg: %f', 1 - correct, cost)

  def print_net_summary(self):
    for s in self.net.get_summary():
      name = s[0]
      values = s[1]
      log("Layer '%s' weight: %e [%e] @ [%e]", name, values[0], values[1], values[4])
      log("Layer '%s' bias: %e [%e] @ [%e]", name, values[2], values[3],values[5])


  def check_test_data(self):
    return self.curr_batch % self.test_freq == 0

  def check_save_checkpoint(self):
    return self.curr_batch % self.save_freq == 0

  def check_adjust_lr(self):
    return self.factor != 1 and self.curr_batch % self.adjust_freq == 0

  def should_continue_training(self):
    return True

  def _finished_training(self):
    dumper = getattr(self, 'train_layer_output_dumper', None)
    if dumper != None:
      cache_outputs(self.net, self.train_dp, dumper, index = -3)
    else:
      util.log('There is no dumper for train data')
    dumper = getattr(self, 'test_layer_output_dumper', None)
    if dumper != None:
      cache_outputs(self.net, self.test_dp, dumper, index = -3)
    else:
      util.log('There is no dumper for test data')

  def train(self, num_epochs=1000):
    self.print_net_summary()
    util.log('Starting training...')

    start_epoch = self.curr_epoch
    last_print_time = time.time()

    min_time = 12
    while (self.curr_epoch - start_epoch <= num_epochs and 
          self.should_continue_training()):
      #if min_time < 1.55:
      #  util.dump_profile()
      #util.dump_profile()
      batch_start = time.time()
      st = time.time()
      train_data = self.train_dp.get_next_batch(self.batch_size)
      #print 'Minibatch fetch:', time.time() - st

      self.curr_epoch = train_data.epoch
      self.curr_batch += 1

      input, label = train_data.data, train_data.labels
      self.net.train_batch(input, label)
      cost, correct, numCase = self.net.get_batch_information()
      self.train_outputs += [({'logprob': [cost, 1 - correct]}, numCase, self.elapsed())]

      if time.time() - last_print_time > 1:
        log('%d.%d: error: %f logreg: %f time: %f', self.curr_epoch, self.curr_batch, 1 - correct, cost, time.time() - batch_start)
        min_time = time.time() - batch_start
        last_print_time = time.time()

      if self.check_test_data():
        self.get_test_error()

      if self.factor != 1.0 and self.check_adjust_lr():
        self.adjust_lr()

      if self.check_save_checkpoint():
        self.save_checkpoint()

    self.get_test_error()
    self.save_checkpoint()
    self.report()
    self._finished_training()

  def report(self):
    rep = self.net.get_report()
    if rep is not None:
      log(rep)

    #import numpy as np
    #fc_time_fprop = np.array(self.net.fc_time_fprop[1:10])
    #fc_time_bprop = np.array(self.net.fc_time_bprop[1:10])
    #conv_time_fprop = np.array(self.net.conv_time_fprop[1:10])
    #conv_time_bprop = np.array(self.net.conv_time_bprop[1:10])
    #print 'fc time fprop', fc_time_fprop, 'average',  np.mean(fc_time_fprop)
    #print 'fc time bprop', fc_time_bprop, 'average',  np.mean(fc_time_bprop)
    #print 'conv time fprop', conv_time_fprop, 'average',  np.mean(conv_time_fprop)
    #print 'conv time bprop', conv_time_bprop, 'average',  np.mean(conv_time_bprop)
    ##timer.dump('timer')

    #from varray import send_data_size, recv_data_size
    #print 'send data', send_data_size
    #print 'recv data', recv_data_size

  @staticmethod
  def get_trainer_by_name(name, param_dict):
    net = FastNet(param_dict['image_shape'])
    if name == 'layerwise':
      param_dict['net'] = net
      return ImageNetLayerwisedTrainer(**param_dict)

    net = FastNet(param_dict['image_shape'])
    load_model(net, param_dict['init_model'])
    param_dict['net'] = net
    if name == 'normal':
      return Trainer(**param_dict)
    elif name == 'minibatch':
      return MiniBatchTrainer(**param_dict)
    else:
      raise Exception, 'No trainer found for name: %s' % name




class MiniBatchTrainer(Trainer):
  def _finish_init(self):
    self.num_epoch = 100000

  def should_continue_training(self):
    return self.curr_batch < self.num_batch


class AutoStopTrainer(Trainer):
  def _finish_init(self):
    auto_stop_alg = getattr(self, 'auto_stop_alg', 'smooth')
    self.scheduler = Scheduler.makeScheduler(auto_stop_alg, self)

  def should_continue_training(self):
    return Trainer.should_continue_training(self) and self.scheduler.should_continue_training()

  def check_save_checkpoint(self):
    return Trainer.check_save_checkpoint(self) and self.scheduler.check_save_checkpoint()


class ImageNetLayerwisedTrainer(Trainer):
  def divide_layers_to_stack(self):
    self.fc_params = []
    self.conv_params = []
    conv = True
    for ld in self.init_model:
      if ld['type'] in ['conv', 'rnorm', 'pool', 'neuron'] and conv:
        # self.conv_params.append(ld)
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
    self.conv_stack = FastNet.split_conv_to_stack(self.conv_params)
    self.fc_stack = FastNet.split_fc_to_stack(self.fc_params)


    self.fc_tmp = [self.fc_stack['fc8'][0], self.softmax_param]
    del self.fc_stack['fc8']
    self.stack = self.fc_stack

    self.initialize_model()

    self.num_epoch = self.frag_epoch
    self.net = FastNet(self.learning_rate, self.image_shape, self.curr_model)

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
      self.test_dp = dp(self.test_layer_output_path, range(count), 'fc')
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
    self.curr_batch = self.curr_epoch = 0
    self.init_replaynet_data_provider()

    model = []
    model.extend(stack)
    model.extend(self.fc_tmp)

    self.train_layer_output_dumper = None
    self.test_layer_output_dumper = None
    size = self.net['fc8'].get_input_size()
    image_shape = ConvDataLayout.get_output_shape(1, 1, size, self.batch_size)
    self.net = FastNet(self.learning_rate, image_shape, model)
    self.replaynet = self.net
    self.num_epoch = self.replaynet_epoch
    Trainer.train(self)

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

  def train(self):
    Trainer.train(self)
    for i, stack in enumerate(self.stack.values()):
      self.train_replaynet(stack)
      self.reset_trainer(i)

      self.net.drop_layer_from('fc8')

      for layer in self.replaynet:
        if layer.type != 'data':
          self.net.append_layer(layer)
      Trainer.train(self)


class ImageNetCatewisedTrainer(MiniBatchTrainer):
  def _finish_init(self):
    assert len(self.num_caterange_list) == len(self.num_batch) and self.num_caterange_list[-1] == 1000
    self.num_batch_list = self.num_batch[1:]
    self.num_batch = self.num_batch[0]

    init_output = self.num_caterange_list[0]
    self.num_caterange_list = self.num_caterange_list[1:]

    fc = self.init_model[-2]
    fc['outputSize'] = init_output

    self.learning_rate_list = self.learning_rate[1:]
    self.learning_rate = self.learning_rate[0]

    self.set_category_range(init_output)
    self.net = FastNet(self.learning_rate, self.image_shape, init_model=self.init_model)
    MiniBatchTrainer._finish_init(self)


  def set_category_range(self, r):
    dp = data.get_by_name(self.data_provider)
    self.train_dp = dp(self.data_dir, self.train_range, category_range=range(r))
    self.test_dp = dp(self.data_dir, self.test_range, category_range=range(r))


  def train(self):
    MiniBatchTrainer.train(self)

    for i, cate in enumerate(self.num_caterange_list):
      self.set_category_range(cate)
      self.curr_batch = self.curr_epoch = 0
      self.num_batch = self.num_batch_list[i]

      model = self.checkpoint_dumper.get_checkpoint()
      layers = model['layers']

      fc = layers[-2]
      fc['weight'] = None
      fc['bias'] = None
      fc['weightIncr'] = None
      fc['biasIncr'] = None
      # for l in layers:
      #  if l['type'] == 'fc':
      #    l['weight'] = None
      #    l['bias'] = None
      #    l['weightIncr'] = None
      #    l['biasIncr'] = None

      # fc = layers[-2]
      fc['outputSize'] = cate

      self.learning_rate = self.learning_rate_list[i]
      self.net = FastNet(self.learning_rate, self.image_shape, init_model=model)

      self.net.clear_weight_incr()
      MiniBatchTrainer.train(self)



class ImageNetCateGroupTrainer(MiniBatchTrainer):
  def _finish_init(self):
    self.num_batch_list = self.num_batch[1:]
    self.num_batch = self.num_batch[0]
    self.learning_rate_list = self.learning_rate[1:]
    self.learning_rate = self.learning_rate[0]

    layers = self.init_model
    fc = layers[-2]
    fc['outputSize'] = self.num_group_list[0]
    self.num_group_list = self.num_group_list[1:]

    self.set_num_group(fc['outputSize'])
    self.net = FastNet(self.learning_rate, self.image_shape, init_model=self.init_model)
    MiniBatchTrainer._finish_init(self)

  def set_num_group(self, n):
    dp = data.get_by_name(self.data_provider)
    self.train_dp = dp(self.data_dir, self.train_range, n)
    self.test_dp = dp(self.data_dir, self.test_range, n)


  def train(self):
    MiniBatchTrainer.train(self)

    for i, group in enumerate(self.num_group_list):
      self.set_num_group(group)
      self.curr_batch = self.curr_epoch = 0
      self.num_batch = self.num_batch_list[i]

      model = self.checkpoint_dumper.get_checkpoint()
      layers = model['layers']

      fc = layers[-2]
      fc['outputSize'] = group
      fc['weight'] = None
      fc['bias'] = None
      fc['weightIncr'] = None
      fc['biasIncr'] = None

      self.learning_rate = self.learning_rate_list[i]
      self.net = FastNet(self.learning_rate, self.image_shape, init_model=model)

      self.net.clear_weight_incr()
      MiniBatchTrainer.train(self)





if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--test_id', help='Test Id', default=None, type=str, required=True)
  parser.add_argument('--data_dir', help='The directory that data stored', required=True)
  parser.add_argument('--param_file', help='The param_file or checkpoint file', required=True)
  parser.add_argument('--data_provider', help='The data provider', choices=['cifar10', 'imagenet', 'imagenetcategroup'], required=True)
  parser.add_argument('--train_range', help='The range of the train set', required=True)
  parser.add_argument('--test_range', help='THe range of the test set', required=True)
  parser.add_argument('--save_freq', help='How often should I save the checkpoint file', default=100, type=int, required=True)
  parser.add_argument('--test_freq', help='How often should I test the model', default=100, type=int, required=True)
  parser.add_argument('--adjust_freq', help='How often should I adjust the learning rate', default=100, type=int)
  parser.add_argument('--factor', help='The factor used to adjust the learning rate', default='1.0')
  parser.add_argument('--learning_rate' , help='The scale learning rate', default='0.1', required=True)
  parser.add_argument('--batch_size', help='The size of batch', default=128, type=int)
  parser.add_argument('--checkpoint_dir', help='The directory to save checkpoint file', required=True)

  parser.add_argument('--trainer', help='The type of the trainer', default='normal', choices=
      ['normal', 'catewise', 'categroup', 'minibatch', 'layerwise'], required=True)

  parser.add_argument('--num_group_list', help='The list of the group you want to split the data to')
  parser.add_argument('--num_caterange_list', help='The list of category range you want to train')
  parser.add_argument('--num_epoch', help='The number of epoch you want to train', default=30, type=int)
  parser.add_argument('--num_batch', help='The number of minibatch you want to train')
  parser.add_argument('--output_dir', help='The directory where to dumper input for last fc layer while training', default='')
  parser.add_argument('--output_method', help='The method to hold the intermediate output', choices=['memory', 'disk'], default='disk')
  parser.add_argument('--replaynet_epoch', help='The #epoch that replaynet(layerwised trainer) will train', default=1, type=int)
  parser.add_argument('--frag_epoch', help='The #epoch that incomplete(layerwised trainer) will train', default=1, type=int)
  parser.add_argument('--multiview', help='Whether use multiview to strenghen the test result', action = 'store_true')

  args = parser.parse_args()


  param_dict = {}
  param_dict['image_color'] = 3
  param_dict['test_id'] = args.test_id

  try:
    int(args.test_id)
    assert False, 'Test id should probably not be an integer anymore.'
  except:
    pass


  param_dict['data_dir'] = args.data_dir
  param_dict['data_provider'] = args.data_provider
  if args.data_provider.startswith('imagenet'):
    param_dict['image_size'] = 224
  elif args.data_provider.startswith('cifar'):
    param_dict['image_size'] = 32
  else:
    assert False, 'Unknown data_provider %s' % args.data_provider

  param_dict['train_range'] = util.string_to_int_list(args.train_range)
  param_dict['test_range'] = util.string_to_int_list(args.test_range)
  param_dict['save_freq'] = args.save_freq
  param_dict['test_freq'] = args.test_freq
  param_dict['adjust_freq'] = args.adjust_freq
  param_dict['multiview'] = args.multiview
  factor = util.string_to_float_list(args.factor)
  if len(factor) == 1:
    param_dict['factor'] = factor[0]
  else:
    param_dict['factor'] = factor


  learning_rate = util.string_to_float_list(args.learning_rate)
  if len(learning_rate) == 1:
    param_dict['learning_rate'] = learning_rate[0]
  else:
    param_dict['learning_rate'] = learning_rate

  param_dict['batch_size'] = args.batch_size
  param_dict['checkpoint_dir'] = args.checkpoint_dir
  trainer = args.trainer


  # create a checkpoint dumper
  image_shape = ConvDataLayout.get_output_shape(param_dict['image_size'], param_dict['image_size'], param_dict['image_color'], param_dict['batch_size'])
  param_dict['image_shape'] = image_shape
  cp_dumper = CheckpointDumper(param_dict['checkpoint_dir'], param_dict['test_id'])
  param_dict['checkpoint_dumper'] = cp_dumper

  # create the init_model
  init_model = cp_dumper.get_checkpoint()
  if init_model is None:
    init_model = parse_config_file(args.param_file)
  param_dict['init_model'] = init_model

  # create train dataprovider and test dataprovider
  dp_class = data.get_by_name(param_dict['data_provider'])
  train_dp = dp_class(param_dict['data_dir'], param_dict['train_range'])
  test_dp = dp_class(param_dict['data_dir'], param_dict['test_range'], multiview = param_dict['multiview'])
  param_dict['train_dp'] = train_dp
  param_dict['test_dp'] = test_dp


  # get all extra information
  num_batch = util.string_to_int_list(args.num_batch)
  if len(num_batch) == 1:
    param_dict['num_batch'] = num_batch[0]
  else:
    param_dict['num_batch'] = num_batch

  param_dict['num_group_list'] = util.string_to_int_list(args.num_group_list)
  param_dict['num_caterange_list'] = util.string_to_int_list(args.num_caterange_list)
  param_dict['output_dir'] = args.output_dir
  param_dict['output_method'] = args.output_method
  param_dict['replaynet_epoch'] = args.replaynet_epoch
  param_dict['frag_epoch'] = args.frag_epoch

  train_layer_output_dumper = None
  test_layer_output_dumper = None
  if param_dict['output_method'] == 'disk':
    if param_dict['output_dir'] != '':
      train_layer_output_path = os.path.join(param_dict['output_dir'], 'train_data.pickle')
      param_dict['train_layer_output_path'] = train_layer_output_path
      train_layer_output_dumper = DataDumper(train_layer_output_path)
      test_layer_output_path = os.path.join(param_dict['output_dir'], 'test_data.pickle')
      param_dict['test_layer_output_path'] = test_layer_output_path
      test_layer_output_dumper = DataDumper(test_layer_output_path)
  elif param_dict['output_method'] == 'memory':
    train_layer_output_dumper = MemoryDataHolder()
    test_layer_output_dumper = MemoryDataHolder()
  param_dict['train_layer_output_dumeper'] = train_layer_output_dumper
  param_dict['test_layer_output_dumeper'] = test_layer_output_dumper

  trainer = Trainer.get_trainer_by_name(trainer, param_dict)
  util.log('start to train...')
  trainer.train(args.num_epoch)
