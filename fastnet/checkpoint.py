from fastnet import util
import cPickle
import glob
import numpy as np
import os
import random
import shelve
import string
import zipfile


class DataDumper(object):
  def __init__(self, target_path, max_mem_size=500e5):
    self.target_path = target_path
    dirname = os.path.dirname(self.target_path)
    if not os.path.exists(dirname):
      os.makedirs(dirname)
      util.log('%s is not exist, create a new directory', dirname)
    self.data = []
    self.sz = 0
    self.count = 0
    self.max_mem_size = max_mem_size

    util.log('dumper establised')
    util.log('target path:    %s', self.target_path)
    util.log('max_memory:     %s', self.max_mem_size)

  def add(self, data):
    for k, v in data.iteritems():
      self.sz += np.prod(v.shape)
    self.data.append(data)

    if self.sz > self.max_mem_size:
      self.flush()

  def flush(self):
    if self.sz == 0:
      return  

    out = {}
    for k in self.data[0].keys():
      items = [d[k] for d in self.data]
      out[k] = np.concatenate(items, axis=0)

    filename = '%s.%d' % (self.target_path, self.count)
    with open(filename, 'w') as f:
      cPickle.dump(out, f, -1)

    util.log('Wrote layer dump to %s', filename)
    self.data = []
    self.sz = 0
    self.count += 1

  def get_count(self):
    return self.count

  def reset(self):
    self.data = []
    self.sz = 0
    self.count = 0

  def get_dir_count(self):
    lis = glob.glob(self.target_path + '.*')
    if lis:
      return len(lis)
    return 0

class MemoryDataHolder(object):
  def __init__(self, single_memory_size=50e6, total_memory_size=4e9):
    self.single_memory_size = single_memory_size
    self.total_memory_size = total_memory_size
    self.single_data_size = 0
    self.total_data_size = 0
    self.count = 0
    self.data = []
    self.memory_chunk = []

    util.log('memory data holder establised')
    util.log('total memory size:    %s', self.total_memory_size)
    util.log('single memory size:   %s', self.single_memory_size)


  def add(self, data):
    for k, v in data.iteritems():
      self.single_data_size += v.nbytes
      self.total_data_size += v.nbytes
    self.data.append(data)

    if self.total_data_size > self.total_memory_size:
      self.cut_off_chunk()

    if self.single_data_size > self.single_memory_size:
      self.flush()


  def flush(self):
    if self.single_data_size == 0:
      return

    dic = {}
    for k in self.data[0].keys():
      items = [d[k] for d in self.data]
      dic[k] = np.concatenate(items, axis=0)

    self.memory_chunk.append(dic)

    util.log('add another memory chunk')
    util.log('memory chunk size:    %s', self.single_data_size)
    util.log('total data size:    %s', self.total_data_size)

    self.data = []
    self.single_data_size = 0
    self.count += 1

  def cut_off_chunk(self):
    if len(self.memory_chunk) == 0:
      util.log('There is no chunk to cut off')
      return

    size = 0
    for k, v, in self.memory_chunk[0].iteritems():
      size += self.memory_chunk[0][k].nbytes

    del self.memory_chunk[0]
    self.total_data_size -= size
    self.count -= 1
    util.log('drop off the first memory chunk')
    util.log('droped chunk size:    %s', size)
    util.log('total data size:      %s', self.total_data_size)

  def finish_push(self):
    self.flush()

  def get_count(self):
    return self.count


class CheckpointDumper(object):
  def __init__(self, checkpoint_dir, test_id, max_cp_size=5e9):
    self.test_id = test_id
    self.counter = iter(xrange(10000))
    self.max_cp_size = max_cp_size
    
    if checkpoint_dir is None:
      util.log_info('Checkpoint directory is None; checkpointing is disabled.')
      self.checkpoint_dir = None
      return
      
    if test_id == '':
      self.checkpoint_dir = checkpoint_dir
    else:
      self.checkpoint_dir = os.path.join(checkpoint_dir, test_id)

    if not os.path.exists(self.checkpoint_dir):
      os.system('mkdir -p \'%s\'' % self.checkpoint_dir)

  def get_checkpoint(self):
    if self.checkpoint_dir is None:
      return None
    
     
    if self.test_id == '':
      cp_pattern = self.checkpoint_dir
    else:
      cp_pattern = os.path.join(self.checkpoint_dir, "*")
    cp_files = glob.glob(cp_pattern)
    if not cp_files:
      return None

    checkpoint_file = sorted(cp_files, key=os.path.getmtime)[-1]
    util.log('Loading from checkpoint file: %s', checkpoint_file)

    try:
      #return shelve.open(checkpoint_file, flag='r', protocol=-1, writeback=False)
      return shelve.open(checkpoint_file, flag='r', protocol=-1, writeback=False)
    except:
      dict = {}
      with zipfile.ZipFile(checkpoint_file) as zf:
        for k in zf.namelist():
          dict[k] = cPickle.loads(zf.read(k))
      return dict

  def dump(self, checkpoint, suffix=0):
    if self.checkpoint_dir is None:
      return
    
    cp_pattern = os.path.join(self.checkpoint_dir, '*')
    cp_files = [(f, os.stat(f)) for f in glob.glob(cp_pattern)]
    cp_files = list(reversed(sorted(cp_files, key=lambda f: f[1].st_mtime)))

    #while sum([f[1].st_size for f in cp_files]) > self.max_cp_size:
    #  os.remove(cp_files.pop())

    checkpoint_filename = "%d" % suffix
    checkpoint_filename = os.path.join(self.checkpoint_dir, checkpoint_filename)

    util.log('Writing checkpoint to %s', checkpoint_filename)
    if checkpoint_filename.startswith('/hdfs'):
      print 'Writing to hdfs '
      suf = ''
      for i in range(6):
        suf += random.choice(string.ascii_letters)
      tempfilename = '/tmp/' + suf
      print 'temp filename is', tempfilename
      sf = shelve.open(tempfilename, flag = 'c', protocol=-1, writeback=False)
      #sf = shelve.open(checkpoint_filename, flag='c', protocol=-1, writeback=False)
      for k, v in checkpoint.iteritems():
        sf[k] = v
      sf.sync()
      sf.close()
      #shutil.copy2(tempfilename, checkpoint_filename)
      os.system('mv %s %s' %( tempfilename, checkpoint_filename))
    else:
      sf = shelve.open(checkpoint_filename, flag='c', protocol=-1, writeback=False)
      for k, v in checkpoint.iteritems():
        sf[k] = v
      sf.sync()
      sf.close()

    util.log('save file finished')

