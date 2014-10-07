import cStringIO as StringIO
import garray
from garray import partial_copy_to, driver
from os.path import basename
from distbase import util, matrix
import Queue
import cPickle
import collections
import glob
import numpy as np
import os
import random
import re
import sys
import threading
import time

from multigpu import arr, rank, num_gpu, multi_gpu

def set_seed():
  seed = arr.get_seed()
  seed = 0
  random.seed(seed)
  np.random.seed(seed)

set_seed()
# determine whether we should copy image data to gpu before training
PREV_FILL_GPU = True
PARALLEL_READ = True

class BatchData(object):
  def __init__(self, data, labels, epoch):
    self.data = data
    self.labels = labels
    self.epoch = epoch

class DataProvider(object):
  def __init__(self, data_dir='.', batch_range=None, minibatch_size = 128):
    self.minibatch_size = minibatch_size
    self.data_dir = data_dir
    self.meta_file = os.path.join(data_dir, 'batches.meta')

    self.curr_batch_index = 0
    self.curr_batch = None
    self.curr_epoch = 1

    if os.path.exists(self.meta_file):
      self.batch_meta = util.load(self.meta_file)
    else:
      util.log_warn('Missing metadata for loader.')

    if batch_range is None:
      self.batch_range = self.get_batch_indexes()
    else:
      self.batch_range = batch_range
    random.shuffle(self.batch_range)

    self.index = 0

  def reset(self):
    self.curr_batch_index = 0
    self.curr_batch = None
    self.curr_epoch = 1
    random.shuffle(self.batch_range)

  def get_next_index(self):
    self.curr_batch_index = self.curr_batch_index + 1
    if self.curr_batch_index == len(self.batch_range) + 1:
      set_seed()
      self.curr_epoch += 1
      self.curr_batch_index = 1

      self._handle_new_epoch()
      random.shuffle(self.batch_range)

    self.curr_batch = self.batch_range[self.curr_batch_index - 1]

  def _handle_new_epoch(self):
    '''
    Called when a new epoch starts.
    '''
    pass

  def get_batch_num(self):
    return len(self.batch_range)

  @property
  def batch_num(self):
    return self.get_batch_num()

  @property
  def image_shape(self):
    return (3, self.inner_size, self.inner_size)

  @property
  def data_dim(self):
    return self.inner_size ** 2 * 3

  def recover_from_dp(self, dp_dict):
    self.curr_batch_index = dp_dict['curr_batch_index']
    self.curr_epoch = dp_dict['curr_epoch']
    self.batch_range = dp_dict['batch_range']
    if self.curr_batch_index < 0:
      self.curr_batch_index = 0

  def dump(self):
    dp = {}
    dp['curr_batch_index'] = self.curr_batch_index - 3
    dp['curr_epoch'] = self.curr_epoch
    dp['batch_range'] = self.batch_range
    return dp

def _prepare_images(data_dir, category_range, batch_range, batch_meta):
  assert os.path.exists(data_dir), data_dir

  dirs = glob.glob(data_dir + '/n*')
  dirs.sort()
  synid_to_dir = {}
  for d in dirs:
    synid_to_dir[basename(d)[1:]] = d

  if category_range is None:
    cat_dirs = dirs
  else:
    cat_dirs = []
    for i in category_range:
      synid = batch_meta['label_to_synid'][i]
      # util.log('Using category: %d, synid: %s, label: %s', i, synid, self.batch_meta['label_names'][i])
      cat_dirs.append(synid_to_dir[synid])

  images = []
  batch_dict = dict((k, k) for k in batch_range)

  for d in cat_dirs:
    imgs = glob.glob(d + '/*.jpg')
    imgs.sort()
    imgs = [v for i, v in enumerate(imgs) if i in batch_dict]
    images.extend(imgs)

  return np.array(images)

class ImageNetDataProvider(DataProvider):
  img_size = 256
  border_size = 16
  inner_size = 224

  def __init__(self, data_dir, batch_range=None, multiview = False, category_range=None, batch_size=1024, minibatch_size = 128):
    DataProvider.__init__(self, data_dir, batch_range, minibatch_size)
    self.multiview = multiview
    self.batch_size = batch_size
    self.images = _prepare_images(data_dir, category_range, batch_range, self.batch_meta)
    self.num_view = 5 * 2 if self.multiview else 1

    assert len(self.images) > 0

    self._shuffle_batches()

    if 'data_mean' in self.batch_meta:
      data_mean = self.batch_meta['data_mean']
    else:
      data_mean = util.load(data_dir + 'image-mean.pickle')['data']

    self.data_mean = (data_mean
        .astype(np.single)
        .T
        .reshape((3, 256, 256))[:,
                                self.border_size:self.border_size + self.inner_size,
                                self.border_size:self.border_size + self.inner_size]
        .reshape((self.data_dim,1))
        )
    util.log('Starting data provider with %d batches', len(self.batches))

  def _shuffle_batches(self):
    # build index vector into 'images' and split into groups of batch-size
    image_index = np.arange(len(self.images))
    np.random.shuffle(image_index)

    self.batches = []
    index = 0
    while index < len(self.images):
      if not multi_gpu:
        self.batches.append(image_index[index: index + self.batch_size])
      else:
        num_images = min(self.batch_size, len(image_index) - index)
        if num_images % num_gpu != 0:
          break
        num_images = util.divup(num_images, num_gpu)
        self.batches.append(image_index[index + rank * num_images: index + (rank+1) * num_images ])
      index += self.batch_size

    self.batch_range = range(len(self.batches))

  def _handle_new_epoch(self):
    self._shuffle_batches()

  def __decode_trim_images1(self, image_filenames, cropped):
    images = []
    for idx, filename in enumerate(image_filenames):
      jpeg = Image.open(filename)
      if jpeg.mode != "RGB": jpeg = jpeg.convert("RGB")
      img = np.asarray(jpeg, np.float32).transpose(2, 0, 1).astype(np.float32)
      images.append(img)

    self.__trim_borders(images, cropped)

  def __trim_borders(self, images, target):
    if self.multiview:
      start_positions = [(0, 0), (0, self.border_size * 2), (self.border_size, self.border_size),
                          (self.border_size *2 , 0), (self.border_size * 2 , self.border_size * 2)]
      end_positions = [(x + self.inner_size, y + self.inner_size) for (x, y) in start_positions]
      for i in xrange(self.num_view / 2):
        startY , startX = start_positions[i][0], start_positions[i][1]
        endY, endX = end_positions[i][0], end_positions[i][1]
        num_image = len(images)
        for idx, img in enumerate(images):
          pic = img[:, startY:endY, startX:endX]
          target[:, :, :, i * num_image + idx] = pic
          target[:, :, :, (self.num_view/2 +i) * num_image + idx] = pic[:, :, ::-1]
    else:
      #matrix.trim_images(images, target, self.img_size, self.border_size)
      for idx, img in enumerate(images):
        startY, startX = np.random.randint(0, self.border_size * 2 + 1), np.random.randint(0, self.border_size * 2 + 1)
        endY, endX = startY + self.inner_size, startX + self.inner_size
        pic = img[:, startY:endY, startX:endX]
        if np.random.randint(2) == 0:
          pic = pic[:, :, ::-1]
        target[:, :, :, idx] = pic

  def __decode_trim_images2(self, image_filenames, cropped):
    images = []
    for filename in image_filenames:
      images.append(open(filename).read())
    matrix.decode_trim_images(images, cropped, self.img_size, self.border_size)

  def get_next_batch(self):
    #start = time.time()
    #util.log_info('Start to get batch from disk')
    self.get_next_index()

    epoch = self.curr_epoch
    batchnum = self.curr_batch
    names = self.images[self.batches[batchnum]]
    num_imgs = len(names)
    labels = np.zeros(len(names))
    cropped = np.ndarray((3, self.inner_size, self.inner_size, num_imgs * self.num_view),
            dtype=np.float32)
    # _load in parallel for training
    self.__decode_trim_images2(names, cropped)

    clabel = []
    # extract label from the filename
    for idx, filename in enumerate(names):
      filename = os.path.basename(filename)
      synid = filename[1:].split('_')[0]
      label = self.batch_meta['synid_to_label'][synid]
      labels[idx] = label

    cropped = np.require(cropped, dtype=np.single, requirements='C')
    old_shape = cropped.shape
    cropped = garray.reshape_last(cropped) - self.data_mean
    cropped = cropped.reshape(old_shape)

    labels = np.array(labels)
    labels = labels.reshape(labels.size,)
    labels = np.require(labels, dtype=np.single, requirements='C')

    #util.log_info('get one batch %f', time.time() - start)
    return BatchData(cropped, labels, epoch)

  def recover_from_dp(self, dp_dict):
    DataProvider.recover_from_dp(self, dp_dict)
    self.batches = dp_dict['batches']
    if 'label_batches' in dp_dict:
      self.label_batches = dp_dict['label_batches']
    else:
      self.label_batches = self.batches

    # recover the image batch information from label batch list
    if len(self.batches[0]) != len(self.label_batches[0]):
      image_batch_size = len(self.label_batches[0]) / num_gpu
      self.batches = []
      for batch in self.label_batches:
        if len(batch) == self.batch_size:
          self.batches.append(batch[rank * image_batch_size: (rank + 1) * image_batch_size].copy())

  def dump(self):
    dp = DataProvider.dump(self)
    dp['batches'] = self.batches
    dp['label_batches'] = self.label_batches
    return dp

class ImageNetBatchDataProvider(DataProvider):
    img_size = 256
    border_size = 16
    inner_size = 224

    def __init__(self, data_dir, batch_range, multiview = False, minibatch_size = 128):
        DataProvider.__init__(self, data_dir, batch_range, minibatch_size)
        self.multiview = multiview
        self.batch_size = self.batch_meta['batch_size']
        self.num_view = 5 * 2 if self.multiview else 1
        data_mean = self.batch_meta['data_mean']
        self.data_mean = (data_mean
            .reshape((3, 256, 256))[:,
                                self.border_size:self.border_size + self.inner_size,
                                self.border_size:self.border_size + self.inner_size]
            .reshape((self.data_dim, 1))
        )
    def __decode_trim_images1(self, data, target):
        images = []
        for raw_data in data:
            file = StringIO.StringIO(raw_data)
            jpeg = Image.open(file)
            images.append(np.asarray(jpeg, np.float32).transpose(2, 0, 1))

        cropped = np.ndarray((3, self.inner_size, self.inner_size, len(images) * self.num_view), dtype = np.float32)
        self.__trim_borders(images, cropped)

    def __trim_borders(self, images, target):
        if self.multiview:
            start_positions = [(0, 0), (0, self.border_size * 2), (self.border_size, self.border_size), (self.border_size *2 , 0), (self.border_size * 2 , self.border_size * 2)]
            end_positions = [(x + self.inner_size, y + self.inner_size) for (x, y) in start_positions]
            for i in xrange(self.num_view / 2):
                startY , startX = start_positions[i][0], start_positions[i][1]
                endY, endX = end_positions[i][0], end_positions[i][1]
                num_image = len(images)
                for idx, img in enumerate(images):
                    pic = img[:, startY:endY, startX:endX]
                    target[:, :, :, i * num_image + idx] = pic
                    target[:, :, :, (self.num_view/2 +i) * num_image + idx] = pic[:, :, ::-1]
        else:
            matrix.trim_images(images, target, self.img_size, self.border_size)

    def __decode_trim_images2(self, images, target):
        matrix.decode_trim_images(images, target, self.img_size, self.border_size)

    def __multigpu_seg(self, data):
        if not multi_gpu:
            return images
        num_images = util.divup(len(images), num_gpu)
        if rank != num_gpu - 1:
            images = data['data'][rank * num_images: (rank + 1) * num_images]
            labels = data['labels'][rank * num_images: (rank + 1) * num_images]
        else:
            images = data['data'][rank * num_images:]
            labels = data['labels'][rank * num_images:]
        data['data'] = images
        data['labels'] = labels
        return data

    def get_next_batch(self):
        self.get_next_index()
        epoch = self.curr_epoch
        filename = os.path.join(self.data_dir, 'data_batch_%d' % (self.curr_batch))
        start = time.time()
        if os.path.isdir(filename):
            images = []
            labels = []

            for sub_filename in os.listdir(filename):
                path = os.path.join(filename, sub_filename)
                data = util.load(path)
                images.extend(data['data'])
                labels.extend(data['labels'])
            data['data'] = images
            data['labels'] = labels
        else:
            data = util.load(filename)
        data = self.__multigpu_seg(data)
        images = data['data']

        cropped = np.ndarray((3, self.inner_size, self.inner_size, len(images) * self.num_view), dtype = np.float32)
        self.__decode_trim_images2(images, cropped)

        cropped = garray.reshape_last(cropped) - self.data_mean
        cropped = np.require(cropped.reshape((3, self.inner_size, self.inner_size, len(images) * self.num_view)), dtype = np.single, requirements='C')

        labels = np.array(labels)
        labels = labels.reshape(labels.size, )
        labels = np.require(labels, dtype=np.single, requirements='C')
        return BatchData(cropped, labels, epoch)

class DummyDataProvider(DataProvider):
  def __init__(self, inner_size, output_size, batch_size = 1024, minibatch_size = 128):
    DataProvider.__init__(self, '/tmp/', batch_range = range(1, 40))
    self.inner_size = inner_size
    self.output_size = output_size
    self.batch_size = batch_size

  def get_next_batch(self):
    batch_size = self.batch_size / num_gpu
    if rank == num_gpu - 1:
      batch_size = self.batch_size - batch_size * (num_gpu -1)

    data = np.ones((3, self.inner_size, self.inner_size, batch_size)).astype(np.float32) * 128
    label = [np.random.choice(self.output_size) for i in range(batch_size)]
    label = np.array(label).astype(np.float32)

    return BatchData(data, label, 1)

class CifarDataProvider(DataProvider):
  img_size = 32
  border_size = 0
  inner_size = 32

  BATCH_REGEX = re.compile('^data_batch_(\d+)$')
  def get_next_batch(self):
    self.get_next_index()
    filename = os.path.join(self.data_dir, 'data_batch_%d' % self.curr_batch)

    data = util.load(filename)
    img = data['data'] - self.batch_meta['data_mean']
    num_image = img.shape[-1]
    label = data['labels']
    if multi_gpu:
      nrow = img.shape[-1] / num_gpu
      pos_from = nrow * rank
      pos_to = (rank+ 1) * nrow

      img = img[:, pos_from : pos_to]
      label = label[pos_from : pos_to]
      num_image = img.shape[-1]
    img_size = CifarDataProvider.img_size
    return BatchData(np.require(img.reshape(3, img_size, img_size, num_image), requirements='C', dtype=np.float32),
                     np.array(label),
                     self.curr_epoch)

  def get_batch_indexes(self):
    names = self.get_batch_filenames()
    return sorted(list(set(int(DataProvider.BATCH_REGEX.match(n).group(1)) for n in names)))


class ReaderThread(threading.Thread):
  def __init__(self, queue, dp):
    threading.Thread.__init__(self)
    self.daemon = True
    self.queue = queue
    self.dp = dp
    self._stop = False
    self._running = True

  def run(self):
    while not self._stop:
      self.queue.put(self.dp.get_next_batch())

    self._running = False

  def stop(self):
    self._stop = True
    while self._running:
      _ = self.queue.get(0.1)

class ParallelDataProvider(DataProvider):
  def __init__(self, dp):
    self.dp = dp
    self._reader = None
    self.reset()

  def _start_read(self):
    util.log('Starting reader...')
    assert self._reader is None
    self._reader = ReaderThread(self._data_queue, self.dp)
    self._reader.start()

  @property
  def image_shape(self):
    return self.dp.image_shape

  @property
  def multiview(self):
    if hasattr(self.dp, 'multiview'):
      return self.dp.multiview
    else:
      return False

  @property
  def batch_size(self):
    if hasattr(self.dp, 'batch_size'):
      return self.dp.batch_size
    else:
      return 0

  @property
  def batch_num(self):
    return self.dp.get_batch_num()

  @property
  def num_view(self):
    if hasattr(self.dp, 'num_view'):
      return self.dp.num_view
    else:
      return 1

  def reset(self):
    self.dp.reset()

    if self._reader is not None:
      self._reader.stop()

    self._reader = None
    self._data_queue = Queue.Queue(1)
    self._gpu_batch = None
    self.index = 0
    self._copy_queue = Queue.Queue(1)
    self._batch_buffer = None

  def _fill_reserved_data(self):
    batch_data = self._data_queue.get()

    self.curr_epoch = batch_data.epoch
    if not self.multiview and PREV_FILL_GPU:
      if self._gpu_batch is not None:
        self._gpu_batch.data.mem_free()
        del self._gpu_batch
      batch_data.data = garray.array(batch_data.data, dtype = np.float32)
      batch_data.labels = garray.array(batch_data.labels, dtype = np.float32)
      self._gpu_batch = batch_data
    else:
      self._cpu_batch = batch_data

  def __get_next_minibatch(self):
    if self.index == 0:
      self._batch_buffer = self._data_queue.get()
    shape = self._batch_buffer.data.shape
    width = shape[-1]
    if self.index + self._batch_size >= width:
      last_dim = width - self.index
      data = driver.pagelocked_empty(shape = shape[:-1] + (last_dim,), dtype = np.float32)
      labels = driver.pagelocked_empty(shape = (last_dim,), dtype = np.float32)
      data[:, :, :, :] = self._batch_buffer.data[:, :, :, self.index:self.index + last_dim]
      labels[:] = self._batch_buffer.labels[self.index:self.index + last_dim]
      self.index = 0
    else:
      data = driver.pagelocked_empty(shape = shape[:-1] + (self._batch_size,), dtype = np.float32)
      labels = driver.pagelocked_empty(shape = (self._batch_size,), dtype = np.float32)
      data[:, :, :, :] = self._batch_buffer.data[:, :, :, self.index:self.index + self._batch_size]
      labels[:] = self._batch_buffer.labels[self.index:self.index + self._batch_size]
      self.index += self._batch_size
    return BatchData(data, labels, self._batch_buffer.epoch)

  def __push_next_minibatch_sync(self):
    batch_data = self.__get_next_minibatch()
    self._copy_queue.put(BatchData(garray.to_gpu(batch_data.data), garray.to_gpu(batch_data.labels), batch_data.epoch))

  def __push_next_minibatch_async(self):
    batch_data = self.__get_next_minibatch()
    self._copy_queue.put(BatchData(garray.to_gpu_async(batch_data.data), garray.to_gpu_async(batch_data.labels), batch_data.epoch))


  def __get_next_batch2(self, batch_size):
    if self._reader is None:
      self._start_read()
    if not hasattr(self, '_batch_size'):
      self._batch_size = batch_size

    if self._copy_queue.empty():
      self.__push_next_minibatch_sync()
    batch_data = self._copy_queue.get()
    self.__push_next_minibatch_async()
    return batch_data

  def __get_next_batch1(self):
    batch_size = self.dp.minibatch_size
    if self._reader is None:
      self._start_read()

    #if self._gpu_batch is None:
    #  self._fill_reserved_data()
    if self.index == 0:
        self._fill_reserved_data()

    # the data has to be compete copy of image data on each worker
    if not self.multiview and PREV_FILL_GPU:
      width = self._gpu_batch.data.shape[-1]
      gpu_data = self._gpu_batch.data
      gpu_labels = self._gpu_batch.labels
      epoch = self._gpu_batch.epoch

      # special case wheh the batch size is equal to the data provider batch size
      label_batch_size = batch_size
      tmp_batch_size = batch_size / num_gpu
      if rank == num_gpu - 1:
        tmp_batch_size = batch_size - tmp_batch_size * (num_gpu -1)
      batch_size = tmp_batch_size

      if batch_size == width:
        data = gpu_data.copy()
        labels = gpu_labels.copy()
        #self._fill_reserved_data()
        self.index = 0
      else:
        if self.index + batch_size >=  width:
          width = width - self.index
          labels = gpu_labels[self.index:self.index + batch_size]
          data = gpu_data[:, :, :, self.index:self.index + batch_size]

          self.index = 0
          #self._fill_reserved_data()
        else:
          labels = gpu_labels[self.index : self.index + batch_size]
          data = gpu_data[:, :, :, self.index:self.index + batch_size]
          self.index += batch_size

      if multi_gpu:
        data = arr.assemble(data)
        #gather label from other workers
        labels = arr.concatenate(labels)
    else:
      channel, img_size, img_size, width = self._cpu_batch.data.shape
      cpu_data = self._cpu_batch.data
      cpu_labels = self._cpu_batch.labels
      epoch = self._cpu_batch.epoch

      width /= self.num_view
      if self.index + batch_size >=  width:
        batch_size = width - self.index

      labels = cpu_labels[self.index:self.index + batch_size]
      data = np.zeros((channel, img_size, img_size, batch_size * self.num_view), dtype = np.float32)
      for i in range(self.num_view):
        data[:, :, :, i* batch_size: (i+ 1) * batch_size] = cpu_data[:, :, :, self.index + width * i : self.index + width * i + batch_size]

      self.index = (self.index + batch_size) / width
      data = uniformed_array(np.require(data, requirements = 'C'), to2dim = True)
      labels = uniformed_array(np.require(labels, requirements = 'C'))

    return BatchData(data, labels, epoch)

  get_next_batch = __get_next_batch1

  def recover_from_dp(self, dp):
    self.index = dp['index']
    self.dp.recover_from_dp(dp)

  def dump(self):
    dp = self.dp.dump()
    dp['index'] = self.index
    return dp

dp_dict = {}
def register_data_provider(name, _class):
  if name in dp_dict:
    print 'Data Provider', name, 'already registered'
  else:
    dp_dict[name] = _class

def get_by_name(name):
  if name not in dp_dict:
    print >> sys.stderr, 'There is no such data provider --', name, '--'
    sys.exit(-1)
  else:
    dp_klass = dp_dict[name]
    def construct_dp(*args, **kw):
      dp = dp_klass(*args, **kw)
      if PARALLEL_READ:
        return ParallelDataProvider(dp)
      else:
        return dp
    return construct_dp

register_data_provider('cifar10', CifarDataProvider)
register_data_provider('dummy', DummyDataProvider)
register_data_provider('imagenet', ImageNetDataProvider)
register_data_provider('imagenet-batch', ImageNetBatchDataProvider)
