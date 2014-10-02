from distbase.state import State, combination_conv, combination_fc
from distbase.state import state0, disw_i, sisw, sidw, sidw_f, disw_b
from distbase.layerdist import LayerDist
from distbase.util import parse_config_file
import sys
import os
import pickle
import argparse

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--model-path', help='Path of model file', required=True)
  parser.add_argument('--desc-path', help='Path of distribution description file, could be generated by predictor', required=True)

  args = parser.parse_args()

  model_path = args.model_path
  desc_path = args.desc_path
  strategy_path = model_path + '.layerdist'
  model = parse_config_file(model_path)
  desc = parse_config_file(desc_path)
  assert len(model) == len(desc)

  layerdists = []
  for desc_dict, layer in zip(desc, model):
    print layer['name'], desc_dict['name']
    assert layer['name'] == desc_dict['name']
    layerdists.append(LayerDist(eval(desc_dict['group_state']), desc_dict['workers_group']))

  strategy = {}
  for i, layer in enumerate(model):
    strategy[layer['name']] = layerdists[i]
    print layer['name'], layerdists[i]

  with open(strategy_path, 'w') as fout:
    pickle.dump(strategy, fout)
