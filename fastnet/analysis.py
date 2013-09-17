#!/usr/bin/env

'''Functions for analyzing the output of fastnet checkpoint files.'''

from fastnet import util
from matplotlib.pyplot import gcf
import cPickle
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas
import pylab
import shelve
import zipfile

def find_latest(pattern):
    import glob
    files = glob.glob(pattern)
    ftimes = sorted((os.stat(f).st_ctime, f) for f in files)
    
    if ftimes:
      return ftimes[-1][1]

    return None

def select(df, **cond):
    k, v = cond.items()[0]
    idx = getattr(df, k) == v

    for k, v in cond.items()[1:]:
        idx2 = getattr(df, k) == v
        idx = idx & idx2

    return df[idx]

def _load_series(data, scale=1):
    lp = [t['logprob'] for t,count,elapsed in data]
    counts = np.array([count for t,count,elapsed in data]).cumsum()
    examples = counts * scale

    elapsed = np.array([elapsed for t,count,elapsed in data])
    logprob = np.array([t[0] for t in lp])
    prec = np.array([t[1] for t in lp])
    return pandas.DataFrame({'lp' : logprob, 'pr' : prec, 'elapsed' : elapsed, 'examples' : examples})

def try_load_zip(state_f):
  try:
    zf = zipfile.ZipFile(state_f, 'r')
    train_outputs = cPickle.loads(zf.read('train_outputs'))
    test_outputs = cPickle.loads(zf.read('test_outputs'))
    return train_outputs, test_outputs
  except:
    return None, None

def try_load_pickle(state_f):
  try:
    data = cPickle.load(state_f)
    train_outputs = data['train_outputs']
    test_outputs = data['test_outputs']
    return train_outputs, test_outputs
  except:
    return None, None

def try_load_shelf(state_f):
  try:
    data = shelve.open(state_f, flag='r')
    train_outputs = data['train_outputs']
    test_outputs = data['test_outputs']
    return train_outputs, test_outputs
  except:
    return None, None
    
def load_checkpoint(pattern):
  state_f = find_latest(pattern)
  train_outputs, test_outputs = try_load_zip(state_f)
  if not train_outputs:
    train_outputs, test_outputs = try_load_pickle(state_f)
  if not train_outputs:
    train_outputs, test_outputs = try_load_shelf(state_f)
    
  assert train_outputs is not None

  train_df = _load_series(train_outputs)
  train_df['type'] = 'train'


  test_df = _load_series(test_outputs)
  test_df['type'] = 'test'

  #return train_df, test_df
  out = pandas.concat([train_df, test_df])
  return out


def plot_df(df, x, y, save_to=None, title=None, merge=False,
            x_label=None, y_label=None, legend=None,
            transform_x=lambda k, x: x, transform_y=lambda k, y: y,
            xlim=None, ylim=None):
    from itertools import cycle
    lines = cycle(["-","--","-.",":"])
    colors = cycle('bgrcmyk')

    if merge: f = gcf()
    else: f = plt.figure()

    if isinstance(df, dict):
        for k in sorted(df.keys()):
            v = df[k]
            ax = f.add_subplot(111)
            ax.plot(transform_x(k, v[x]), transform_y(k, v[y]),
                linestyle=lines.next(), color=colors.next(), label='%s' % k)
    else:
        ax = f.add_subplot(111)
        ax.plot(df[x], df[y], linestyle=lines.next(), color=colors.next())

    ax.set_title(title)
    if legend: ax.legend(title=legend)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if xlim is not None: ax.set_xlim(xlim)
    if ylim is not None: ax.set_ylim(ylim)
    #ax.set_yscale('log')

    f.set_figheight(8)
    f.set_figwidth(12)
    if save_to is not None:
        pylab.savefig(save_to, bbox_inches=0)


def plot_series(frame, groupby, x, y, **kw):
    g = frame.groupby(groupby)
    df = dict([(k, g.get_group(k)) for k in g.groups.keys()])
    kw['x_label'] = x
    kw['y_label'] = y
    plot_df(df, x, y, **kw)

def build_image(array):
  if len(array.shape) == 4:
    filter_size = array.shape[1]
  else:
    filter_size = array.shape[0]
  
  num_filters = array.shape[-1]
  num_cols = util.divup(80, filter_size)
  num_rows = util.divup(num_filters, num_cols)

  if len(array.shape) == 4:
    big_pic = np.zeros((3, (filter_size + 1) * num_rows, (filter_size + 1) * num_cols))
  else:
    big_pic = np.zeros((filter_size * num_rows, filter_size * num_cols))
  
  for i in range(num_rows):
    for j in range(num_cols):
      idx = i * num_cols + j
      if idx >= num_filters: break
      x = i*(filter_size + 1)
      y = j*(filter_size + 1)
      if len(array.shape) == 4:
        big_pic[:, x:x+filter_size, y:y+filter_size] = array[:, :, :, idx]
      else:
        big_pic[x:x+filter_size, y:y+filter_size] = array[:, :, idx]
  
  if len(array.shape) == 4:
    return big_pic.transpose(1, 2, 0)
  return big_pic

def load_layer(f, layer_id=1):
  cp = find_latest(f)
  try:
    sf = shelve.open(cp, flag='r')
    layer = sf['layers'][layer_id]
    imgs = layer['weight']
    filters = layer['numFilter']
    filter_size = layer['filterSize']
  except:
    zf = zipfile.ZipFile(cp)
    layer = cPickle.loads(zf.read('layers'))[layer_id]
    imgs = layer['weight']
    filters = layer['numFilter']
    filter_size = layer['filterSize']
  
  imgs = imgs.reshape(3, filter_size, filter_size, filters)
  return imgs

def plot_filters(imgs):
  imgs = imgs - imgs.min()
  imgs = imgs / imgs.max()
  fig = pylab.gcf()
  fig.set_size_inches(12, 8)
  
  ax = fig.add_subplot(111)
  big_pic = build_image(imgs)
  ax.imshow(big_pic, interpolation='nearest')
  return imgs

def plot_file(f, layer_id=1):
  return plot_filters(load_layer(f, layer_id))
  
  
def diff_files(a, b):
  f_a = load_layer(a)
  f_b = load_layer(b)
  fig = pylab.gcf()
  fig.set_size_inches(12, 8)
  
  ax = fig.add_subplot(111)
  diff = np.abs(f_a - f_b)
  #diff = diff - diff.min()
  #diff = diff / diff.max()
  #big_pic = build_image(diff)
  #print diff[0, :, :, 0]
  #print f_a[0, :, :, 0]
  #print f_b[0, :, :, 0]
  diff = diff / max(np.max(f_a), np.max(f_b))
  ax.imshow(build_image(diff))
