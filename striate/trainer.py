from fastnet import *
from data import DataProvider, ParallelDataProvider
from options import *
import cPickle
from util import timer
import time
import re

def load(filename):
  with open(filename, 'rb') as f:
    model = cPickle.load(f)
  return model


class Trainer:
  CHECKPOINT_REGEX = None
  def __init__(self, test_id, data_dir, checkpoint_dir, train_range, test_range, test_freq,
      save_freq, batch_size, num_epoch, image_size, image_color, learning_rate, n_out, autoInit=True):
    self.test_id = test_id
    self.data_dir = data_dir
    self.checkpoint_dir = checkpoint_dir
    self.train_range = train_range
    self.test_range = test_range
    self.test_freq = test_freq
    self.save_freq = save_freq
    self.batch_size = batch_size
    self.num_epoch = num_epoch
    self.image_size = image_size
    self.image_color = image_color
    self.learning_rate = learning_rate
    self.n_out = n_out
    self.regex = re.compile('^test%d-(\d+)\.(\d+)$' % self.test_id)

    self.train_dp = DataProvider(self.batch_size, self.data_dir, self.train_range)
    self.test_dp = DataProvider(self.batch_size, self.data_dir, self.test_range)

    self.image_shape = (self.batch_size, self.image_color, self.image_size, self.image_size)
    self.train_outputs = []
    self.test_outputs = []
    self.net = FastNet(self.learning_rate, self.image_shape, self.n_out, autoAdd = autoInit)

    self.num_batch = self.curr_epoch = self.curr_batch = 0
    self.train_data = None
    self.test_data = None

    self.num_train_minibatch = 0
    self.num_test_minibatch = 0
    self.checkpoint_file = ''

  def get_next_minibatch(self, i, train = TRAIN):
    if train == TRAIN:
      num = self.num_train_minibatch
      data = self.train_data
    else:
      num = self.num_test_minibatch
      data = self.test_data

    batch_data = data['data']
    batch_label = data['labels']
    batch_size = self.batch_size

    if i == num -1:
      input = batch_data[:, i * batch_size: -1]
      label = batch_label[i* batch_size : -1]
    else:
      input = batch_data[:, i * batch_size: (i +1)* batch_size]
      label = batch_label[i * batch_size: (i + 1) * batch_size]

    return input, label


  def save_checkpoint(self):
    model = {}
    model['batchnum'] = self.train_dp.get_batch_num()
    model['epoch'] = self.num_epoch + 1
    model['layers'] = self.net.get_dumped_layers()
    model['train_outputs'] = self.train_outputs
    model['test_outputs'] = self.test_outputs

    dic = {'model_state': model, 'op':None}
    saved_filename = [f for f in os.listdir(self.checkpoint_dir) if self.regex.match(f)]
    for f in saved_filename:
      os.remove(os.path.join(self.checkpoint_dir, f))
    checkpoint_filename = "test%d-%d.%d" % (self.test_id, self.curr_epoch, self.curr_batch)
    checkpoint_file_path = os.path.join(self.checkpoint_dir, checkpoint_filename)
    self.checkpoint_file = checkpoint_file_path
    print checkpoint_file_path
    with open(checkpoint_file_path, 'w') as f:
      cPickle.dump(dic, f)

  def get_test_error(self):
    start = time.time()
    _, _, self.test_data = self.test_dp.get_next_batch()

    self.num_test_minibatch = ceil(self.test_data['data'].shape[1], self.batch_size)
    for i in range(self.num_test_minibatch):
      input, label = self.get_next_minibatch(i, TEST)
      label = np.array(label).astype(np.float32)
      label.shape = (label.size, 1)
      self.net.train_batch(input, label, TEST)
    cost , correct, numCase = self.net.get_batch_information()
    self.test_outputs += [({'logprob': [cost, 1-correct]}, numCase, time.time() - start)]
    print 'error: %f logreg: %f time: %f' % (1-correct, cost, time.time() -
      start)


  def train(self):
    self.curr_epoch, self.curr_batch, self.train_data = self.train_dp.get_next_batch()#self.train_dp.wait()
    while self.curr_epoch <= self.num_epoch:
      start = time.time()
      self.num_train_minibatch = ceil(self.train_data['data'].shape[1], self.batch_size)

      # when loading data, training at the same time


      for i in range(self.num_train_minibatch):
        input, label = self.get_next_minibatch(i)
        label = np.array(label).astype(np.float32)
        label.shape = (label.size, 1)
        self.net.train_batch(input, label)

      cost , correct, numCase = self.net.get_batch_information()
      self.train_outputs += [({'logprob': [cost, 1-correct]}, numCase, time.time() - start)]
      print '%d.%d: error: %f logreg: %f time: %f' % (self.curr_epoch, self.curr_batch, 1-correct, cost, time.time() -
          start)

      self.num_batch += 1
      if self.num_batch % self.test_freq == 0:
        print '---- test ----'
        self.get_test_error()
        print '------------'

      if self.num_batch % self.save_freq == 0:
        print '---- save checkpoint ----'
        self.save_checkpoint()
        print '------------'

      self.curr_epoch, self.curr_batch, self.train_data = self.train_dp.get_next_batch()#self.train_dp.wait()

    if self.num_batch % self.save_freq != 0:
      print '---- save checkpoint ----'
      self.save_checkpoint()




class LayerwisedTrainer(Trainer):
  def __init__(self, test_id, data_dir, checkpoint_dir, train_range, test_range, test_freq,
      save_freq, batch_size, num_epoch, image_size, image_color, learning_rate, n_filters,
      size_filters, fc_nouts):
    Trainer.__init__(self, test_id, data_dir, checkpoint_dir, train_range, test_range, test_freq,
        save_freq, batch_size, num_epoch, image_size, image_color, learning_rate, 0, False)
    if len(n_filters) == 1:
      self.layerwised = False
    else:
      self.layerwised = True


    self.n_filters = n_filters
    self.size_filters = size_filters
    self.fc_nouts = fc_nouts

    init_n_filter = [self.n_filters[0]]
    init_size_filter = [self.size_filters[0]]
    

    self.net.add_parameterized_layers(init_n_filter, init_size_filter, self.fc_nouts)

  def train(self):
    if self.layerwised :
      print 'Train the first conv-pool-rnorm stack'
    
    Trainer.train(self)
    
    if self.layerwised:
      for i in range(len(self.n_filters) - 1):
        next_n_filter = [self.n_filters[i +1]]
        next_size_filter = [self.size_filter[i+1]]
        model = load(self.checkpoint_file)
        self.net = FastNet(self.learning_rate, self.image_shape, model = model)
        self.del_layer()
        self.del_layer()
        self.net.disable_bprop()
        
        self.add_parameterized_layers(next_n_filter, next_size_filter, self.fc_nouts)
        
        Trainer.train(self)
    
   

if __name__ == '__main__':
  test_id = 1
  data_dir = '/hdfs/cifar/data/cifar-10-python'
  checkpoint_dir = './checkpoint/'
  train_range = range(1, 41)
  test_range = range(41, 49)

  test_freq = 5
  save_freq = 10
  batch_size = 128
  num_epoch = 2

  image_size = 32
  image_color = 3
  learning_rate = 0.64
  n_filters = [64, 64]
  size_filters = [5, 5]
  fc_nouts = [10]
  trainer = LayerwisedTrainer(test_id, data_dir, checkpoint_dir, train_range, test_range, test_freq,
      save_freq, batch_size, num_epoch, image_size, image_color, learning_rate, n_filters,
      size_filters, fc_nouts)

  trainer.train()

  timer.report()