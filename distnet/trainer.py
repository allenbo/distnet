from distbase import util
from distbase.util import log, timer
from distbase.monitor import MONITOR
from distnet import argparse, layer, data
from collections import deque
from distnet.checkpoint import CheckpointDumper, DataDumper, MemoryDataHolder
from distnet.layer import TRAIN, TEST, backend_name
from distnet.net import FastNet
from distnet.lr import Stat
from distnet.parser import parse_config_file, load_model
from distnet.scheduler import Scheduler
import os
import sys
import time
import math
import garray
import numpy as np
from garray import ConvDataLayout, GPUArray

# Trainer should take: (training dp, test dp, distnet, checkpoint dir)
class Trainer:
  def __init__(self, checkpoint_dumper, train_dp, test_dp, batch_size, num_epochs, net=None, **kw):
    self.checkpoint_dumper = checkpoint_dumper
    self.train_dp = train_dp
    self.test_dp = test_dp
    self.batch_size = batch_size
    self.num_epochs = num_epochs
    self.net = net
    self.start_time = time.time()
    self.train_outputs = []
    self.test_outputs = []
    self.base_time = 0

    for k, v in kw.iteritems():
      setattr(self, k, v)

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
    util.log_info('%s', self.stat)
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
    test_data = self.test_dp.get_next_batch()

    input, label = test_data.data, test_data.labels
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

  def train(self):
    #self.print_net_summary()
    util.log('Starting training...')

    start_epoch = self.stat.curr_epoch
    last_print_time = time.time()

    min_time = 12
    while (self.stat.curr_epoch - start_epoch <= self.num_epochs and self.should_continue_training()):
      batch_start = time.time()
      train_data = self.train_dp.get_next_batch()

      self.stat.curr_epoch = train_data.epoch
      self.stat.curr_batch += 1

      input, label = train_data.data, train_data.labels

      if isinstance(input, np.ndarray):
        input = garray.array(input)

      if isinstance(label, np.ndarray):
        label = garray.array(label.reshape((1, label.size)))

      self.net.train_batch(input, label, self.stat)
      cost, correct, numCase = self.net.get_batch_information()
      self.train_outputs += [({'logprob': [cost, 1 - correct]}, numCase, self.elapsed())]

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

  def predict(self):
    util.log('Starting training...')
    start_epoch = -1
    total_cost, total_correct, total_case = 0, 0, 0
    while True:
      batch_size = self.batch_size
      test_data = self.test_dp.get_next_batch()

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
