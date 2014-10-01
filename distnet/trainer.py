from distbase import util
from distbase.util import log, timer
from distbase.monitor import MONITOR
from distnet import argparse, layer, data
from collections import deque
from distnet.data import PARALLEL_READ
from distnet.checkpoint import CheckpointDumper, DataDumper, MemoryDataHolder
from distnet.layer import TRAIN, TEST, backend_name
from distnet.net import FastNet
from distnet.lr import Stat
from distnet.parser import parse_config_file, load_model
from distnet.scheduler import Scheduler
from distnet.multigpu import init_strategy
import os
import sys
import time
import math
import garray
import numpy as np
from garray import ConvDataLayout, GPUArray


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
  def __init__(self, checkpoint_dumper, train_dp, test_dp, batch_size, num_epochs, net=None, **kw):
    self.checkpoint_dumper = checkpoint_dumper
    self.train_dp = train_dp
    self.test_dp = test_dp
    self.batch_size = batch_size
    self.num_epochs = num_epochs
    self.net = net
    self.multiview = False
    self.start_time = time.time()
    self.train_outputs = []
    self.test_outputs = []
    self.base_time = 0

    for k, v in kw.iteritems():
      setattr(self, k, v)
    init_strategy(dist_file = self.param_file + '.layerdist')

    # recover from checkpoint
    checkpoint = self.checkpoint_dumper.get_checkpoint()
    if checkpoint and len(checkpoint['train_outputs']) != 0:
      self.train_outputs = checkpoint['train_outputs']
      self.test_outputs = checkpoint['test_outputs']
      self.base_time = self.train_outputs[-1][-1]

      self.train_dp.recover_from_dp(checkpoint['train_dp'])
      self.test_dp.recover_from_dp(checkpoint['test_dp'])

      self.num_epochs = checkpoint.get('num_epochs', num_epochs)
      self.stat = checkpoint.get('stat', None)

    if not hasattr(self, 'stat') or self.stat is None:
      total_batch = self.num_epochs * (self.train_dp.batch_size  / self.batch_size) * self.train_dp.batch_num
      self.stat = Stat(self.num_epochs, total_batch, 0, 0, self.batch_size)
    print self.stat
    self._finish_init()

  def _finish_init(self):
    pass

  def init_data_provider(self):
    self.train_dp.reset()
    self.test_dp.reset()

  def save_checkpoint(self):
    model = {}
    model['layers'] = self.net.get_dumped_layers()
    model['train_outputs'] = self.train_outputs
    model['test_outputs'] = self.test_outputs
    model['num_epochs'] = self.num_epochs
    model['stat'] = self.stat
    model['train_dp'] = self.train_dp.dump()
    model['test_dp'] = self.test_dp.dump()
    model['backend'] = backend_name

    log('---- save checkpoint ----')
    #self.print_net_summary()
    self.checkpoint_dumper.dump(checkpoint=model, suffix=self.stat.curr_epoch)


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
      self.net.train_batch(input, label, self.stat, TEST)
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
    return self.stat.curr_batch % self.test_freq == 0

  def check_save_checkpoint(self):
    return self.stat.curr_batch % self.save_freq == 0

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

  def train(self):
    #self.print_net_summary()
    util.log('Starting training...')

    start_epoch = self.stat.curr_epoch
    last_print_time = time.time()

    min_time = 12
    while (self.stat.curr_epoch - start_epoch <= self.num_epochs and self.should_continue_training()):
      #util.dump_profile()
      if PARALLEL_READ == True:
        batch_start = time.time()
      #start = time.time()
      train_data = self.train_dp.get_next_batch(self.batch_size)
      #util.log_info('Trainer get data %f', time.time() - start)

      self.stat.curr_epoch = train_data.epoch
      self.stat.curr_batch += 1

      input, label = train_data.data, train_data.labels

      if isinstance(input, np.ndarray):
        input = garray.array(input)

      if isinstance(label, np.ndarray):
        label = garray.array(label.reshape((1, label.size)))

      if PARALLEL_READ == False:
        batch_start = time.time()
      #start = time.time()
      self.net.train_batch(input, label, self.stat)
      cost, correct, numCase = self.net.get_batch_information()
      self.train_outputs += [({'logprob': [cost, 1 - correct]}, numCase, self.elapsed())]
      #util.log_info('Trainer train one minibatch %f', time.time() - start)
      if time.time() - last_print_time > 0:
        log('%d.%d: error: %f logreg: %f time: %f', self.stat.curr_epoch, self.stat.curr_batch, 1 - correct, cost, time.time() - batch_start)
        MONITOR.report()
        self.net.batch_report()
        min_time = time.time() - batch_start
        last_print_time = time.time()

      if self.check_test_data():
        self.get_test_error()

      if self.check_save_checkpoint():
        self.save_checkpoint()

    self.get_test_error()
    self.save_checkpoint()
    self.report()
    self._finished_training()

  def predict(self):
    util.log('Starting training...')
    start_epoch = -1
    total_cost, total_correct, total_case = 0, 0, 0
    while True:
      batch_size = self.batch_size
      test_data = self.test_dp.get_next_batch(batch_size)

      self.stat.curr_batch += 1
      self.stat.curr_epoch = test_data.epoch
      if start_epoch == -1:
        start_epoch = self.stat.curr_epoch
      if start_epoch != self.stat.curr_epoch:
        break
      input, label = test_data.data, test_data.labels
      self.net.train_batch(input, label, TEST)
      cost , correct, numCase, = self.net.get_batch_information()
      total_cost += cost
      total_correct += correct * numCase
      total_case += numCase
      log('current batch: %d, error rate: %f, overall error rate: %f', self.stat.curr_batch, 1 - correct,
          1 - total_correct / total_case)

  def report(self):
    rep = self.net.get_report()
    if rep is not None:
      log(rep)

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
  param_dict['param_file'] = args.param_file

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
