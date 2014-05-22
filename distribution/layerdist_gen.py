import reader
from distbase.state import State, combination_conv, combination_fc, state0, disw_i, sisw, sidw, sidw_f, disw_b
from distbase.layerdist import LayerDist
import pickle


conv_image_dist = LayerDist(False, disw_i, [2])
fc_shared_dist = LayerDist(False, sisw, [2])


model_path = '../config/cifar-13pct.cfg'
strategy_path = model_path + '.layerdist'
model = reader.getmodel(model_path)

layerdists = [conv_image_dist] * (len(model) - 2) + [fc_shared_dist] * 2

assert len(model) == len(layerdists)

strategy = {}
for i, layer in enumerate(model):
  strategy[layer['name']] = layerdists[i]

with open(strategy_path, 'w') as fout:
    pickle.dump(strategy, fout)
