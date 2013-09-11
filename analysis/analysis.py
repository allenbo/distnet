#!/usr/bin/env

'''Functions for analyzing the output of fastnet checkpoint files.'''

import cPickle
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas
import zipfile
import shelve

def find_latest(pattern):
    import glob
    files = glob.glob(pattern)
    ftimes = sorted((os.stat(f).st_mtime, f) for f in files)
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
    train_outputs = cPickle.load(zf.open('train_outputs'))
    test_outputs = cPickle.load(zf.open('test_outputs'))
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

  out = pandas.merge(train_df, test_df, how='outer')
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

