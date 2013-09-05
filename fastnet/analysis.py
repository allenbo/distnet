#!/usr/bin/env

'''Functions for analyzing the output of fastnet checkpoint files.'''

import cPickle
import numpy as np
import os
import pandas
import zipfile

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

def load_checkpoint(pattern):
    state_f = find_latest(pattern)
    try:
      zf = zipfile.ZipFile(state_f, 'r')
      train_outputs = cPickle.load(zf.open('train_outputs'))
      test_outputs = cPickle.load(zf.open('train_outputs'))
    except:
      data = cPickle.load(state_f)
      train_outputs = data['train_outputs']
      test_outputs = data['test_outputs']
      
      
    train_df = _load_series(train_outputs)
    train_df['type'] = 'train'
    
    test_df = _load_series(test_outputs)

    test_df['type'] = 'test'
    return pandas.merge(train_df, test_df, how='outer')