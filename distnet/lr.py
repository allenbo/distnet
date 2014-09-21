import re
from distbase import util

class Stat(object):
  def __init__(self, total_epoch, total_batch, curr_epoch, curr_batch, batch_size):
    self.total_epoch = total_epoch
    self.total_batch = total_batch
    self.curr_epoch = curr_epoch
    self.curr_batch = curr_batch
    self.batch_size = batch_size

  @property
  def prog(self):
    return self.curr_batch * 1.0 / self.total_batch

  def __str__(self):
    return 'total_epoch:%d total_batch:%d curr_epoch:%d curr_batch:%d batch_size:%d' % (self.total_epoch, self.total_batch, self.curr_epoch, self.curr_batch, self.batch_size)

class LearningRate(object):
  def __init__(self):
    pass

  def get_value(self, stat):
    assert False, 'Not implemented'

  @staticmethod
  def build_learning_rate(desc):
    if isinstance(desc, type('')):
      if desc.startswith('DiscreteExp'):
        match = re.search(r'base=([\d\.]+);factor=([\d\.]+);step=(\d+)', desc)
        if match:
          base = float(match.group(1))
          factor = float(match.group(2))
          step = int(match.group(3))
          return DiscreteExpLearningRate(base, factor, step)
      util.log_fatal('Incorrect definition on learning rate ' + desc)
    elif isinstance(desc, LearningRate):
      return desc
    else:
      return ConstantLearningRate(desc)

class ConstantLearningRate(LearningRate):
  def __init__(self, value):
    LearningRate.__init__(self)
    self.value = value

  def get_value(self, stat):
    return self.value

  def __str__(self):
    return 'Constant[' + str(self.value) + ']'


class DiscreteExpLearningRate(LearningRate):
  def __init__(self, base, factor, step):
    LearningRate.__init__(self)
    self.base = base
    self.factor = factor
    self.step = step
    self.lrs = []
    self.progresses = []
    for i in range(self.step):
      self.lrs.append(self.base * (self.factor ** i))
      self.progresses.append((i+1) * 1.0 / self.step)

  def get_value(self, stat):
    p = stat.prog
    for i, prog in enumerate(self.progresses):
      if p < prog:
        return self.lrs[i]

  def __str__(self):
    return 'DiscreteExp[base:%f,factor:%f,step:%d]' % (self.base, self.factor, self.step)
