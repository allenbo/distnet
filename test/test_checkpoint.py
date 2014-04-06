from distnet import net, layer, checkpoint, parser
import distnet
import garray
import numpy as np

garray.device_init()

def _make_simple():
  net1 = net.FastNet((3, 32, 32))
  net1.append_layer(layer.DataLayer('data', (3, 32, 32)))
  net1.append_layer(layer.ConvLayer('conv1', 32, (5, 5), 1, 1))  
  return net1

def test_checkpoint():
  net1 = _make_simple()
  cp = checkpoint.CheckpointDumper('/tmp/', 'cp_test')
  
  model = {}
  model['layers'] = net1.get_dumped_layers()
  cp.dump(model)
  
  net2 = parser.load_model(net.FastNet((3, 32, 32)), cp.get_checkpoint())
  
  for l1, l2 in zip(net1, net2):
    assert l1.name == l2.name, (l1.name, l2.name)
    if hasattr(l1, 'weight'):
      assert np.all((l1.weight.wt == l2.weight.wt).get())
      assert np.all((l1.bias.wt == l2.bias.wt).get())
