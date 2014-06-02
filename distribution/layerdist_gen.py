import reader
from distbase.state import State, combination_conv, combination_fc, state0, disw_i, sisw, sidw, sidw_f, disw_b
from distbase.layerdist import LayerDist
import pickle


conv_image_dist = LayerDist(disw_i, [2, 2])
conv_batch_dist = LayerDist(disw_b, [2, 2])
fc_shared_dist = LayerDist(sisw, [4])
fc_first_dist = LayerDist(sidw_f, [4])
fc_batch_dist = LayerDist(disw_b, [2])


model_path = '../config/imagenet.cfg'
strategy_path = model_path + '.layerdist'
model = reader.getmodel(model_path)

# image distribution for cifar 13
#layerdists = [conv_image_dist] * 6 + [fc_shared_dist] * 2
# batch_distribution for cifar 13
#layerdists = [conv_batch_dist] * 6 + [fc_shared_dist] * 2
# mix distribution for cifar 13, batch-image
#layerdists = [conv_batch_dist] * 3 + [conv_image_dist] * 3 + [fc_shared_dist] * 2
# mix distribution for cifar 13, image-batch
#layerdists = [conv_image_dist] * 3 + [conv_batch_dist] * 3 + [fc_shared_dist] * 2

# image distribution for cifar 18
#layerdists = [conv_image_dist] * 11 + [fc_shared_dist] * 2
# batch distribution for cifar 18
#layerdists = [conv_batch_dist] * 11 + [fc_shared_dist] * 2
# mix for cifar 18, batch-image-batch
#layerdists = [conv_batch_dist] * 4 + [conv_image_dist] * 4 + [conv_batch_dist] * 3 + [fc_shared_dist] * 2
# mix for cifar 18, image-batch-image
#layerdists = [conv_image_dist] * 4 + [conv_batch_dist] * 4 + [conv_image_dist] * 3 + [fc_shared_dist] * 2


# image distribution with distributed fc layer of cifar 13
#layerdists = [conv_image_dist] * 6 + [fc_batch_dist] * 2


#image distribution, batch on conv layers, sidw_f on fc layers
#layerdists = [conv_batch_dist] *  (len(model) - 6) +  [fc_first_dist] * 4 + [fc_shared_dist] * 2

#2 groups on  image distribution
layerdists = [conv_image_dist] * (len(model) - 6) +  [fc_first_dist] * 4 + [fc_shared_dist] * 2

assert len(model) == len(layerdists)

strategy = {}
for i, layer in enumerate(model):
  strategy[layer['name']] = layerdists[i]
  print layer['name'], layerdists[i]

with open(strategy_path, 'w') as fout:
  pickle.dump(strategy, fout)
