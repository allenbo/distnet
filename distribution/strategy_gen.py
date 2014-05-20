import reader
from distbase.state import State, combination_conv, combination_fc, state0, disw_i, sisw, sidw, sidw_f, disw_b

import pickle
# mix for imagenet
#states = [disw_i] * 4 + [disw_b] * 4 + [disw_i] * (len(model) - 6 - 8) + [sisw] * 6
# image for imagenet
#states = [disw_i] * (len(model) - 6) + [sisw] * 6
# filter for imagenet
#states = [sidw] * (len(model) - 6) + [sisw] * 6
# batch for imagenet
#states = [disw_b] * (len(model) - 6) + [sisw] * 6
# mix for cifar, batch-image
#states = [disw_b] * 3 + [disw_i] * 3 + [sisw] * 2
# mix for cifar, image-batch
#states = [disw_i] * 3 + [disw_b] * 3 + [sisw] * 2
# mix for cifar, 18 batch-image-batch
#states = [disw_b] * 4 + [disw_i] * 4 + [disw_b] * 3 + [sisw] * 2
# mix for cifar, 18 image-batch-image
states = [disw_i] * 4 + [disw_b] * 4 + [disw_i] * 3 + [sisw] * 2

model_path = '../config/imagenet.cfg'
strategy_path = model_path + '.strategy'
model = reader.getmodel(model_path)
assert len(model) == len(states)

strategy = {}
for i, layer in enumerate(model):
  strategy[layer['name']] = states[i]

with open(strategy_path, 'w') as fout:
    pickle.dump(strategy, fout)
