from .util import issquare
class State(object):
  dist   = 'dist'
  dist_b = 'dist-by-batch'
  dist_i = 'dist-by-image'
  dist_f = 'dist-by-first'
  shared = 'shared'


state0 = (0, 0)
sisw = (State.shared, State.shared)
#for conv layer
sidw = (State.shared, State.dist)
disw_i = (State.dist_i, State.shared)
#for fc layer
sidw_f = (State.shared, State.dist_f)
#for both
disw_b = (State.dist_b, State.shared)

#combination_conv =(sisw, sidw, disw_b, disw_i)
#combination_fc = (sisw, sidw_f, disw_b)
combination_conv =(sisw, disw_b, disw_i)
combination_fc = (sisw, sidw_f, disw_b)

def get_input_slice_dim(state, conv, ConvDataLayout, FCDataLayout, num_worker):
  if state is None:
    return None

  if state[0] == State.shared:
    return None

  if state == disw_b:
    if conv:
      return ConvDataLayout.BATCH
    else:
      return FCDataLayout.BATCH

  if state == disw_i:
    assert conv
    if issquare(num_worker):
      return (ConvDataLayout.HEIGHT, ConvDataLayout.WIDTH)
    else:
      return ConvDataLayout.HEIGHT

def get_output_slice_dim(state, conv, ConvDataLayout, FCDataLayout, num_worker):
  if state is None:
    return None

  if state == sisw:
    return None

  if state == sidw:
    assert conv
    return ConvDataLayout.CHANNEL

  if state == disw_i:
    assert conv
    if issquare(num_worker):
      return (ConvDataLayout.HEIGHT, ConvDataLayout.WIDTH)
    else:
      return ConvDataLayout.HEIGHT

  if state == disw_b:
    if conv:
      return ConvDataLayout.BATCH
    else:
      return FCDataLayout.BATCH
  
  if state == sidw_f:
    assert not conv
    return FCDataLayout.NEURON

  assert False

def get_weight_slice_dim(state, conv, FilterLayout, WeightLayout):
  if state == None or state == sisw:
    return None

  if state == sidw:
    assert conv
    return FilterLayout.NUM

  if state == disw_i:
    assert conv
    return None

  if state == disw_b:
    return None

  if state == sidw_f:
    assert not conv
    return WeightLayout.OUTPUT

def get_bias_slice_dim(state, conv):
  if state == sidw:
    assert conv
    return 0 # bias has to be distributed
  if state == sidw_f:
    assert not conv
    return 0

  return None

def get_state_from_slice_dim(output_dist, conv, ConvDataLayout, FCDataLayout):
  if conv:
    if output_dist == ConvDataLayout.BATCH:
      return disw_b
    elif output_dist == (ConvDataLayout.HEIGHT, ConvDataLayout.WIDTH) or output_dist == ConvDataLayout.HEIGHT:
      return disw_i
    elif output_dist is None:
      return sisw
    elif output_dist == ConvDataLayout.CHANNEL:
      return sidw
    else:
      assert False
  else:
    if output_dist is None:
      return sisw
    elif output_dist == FCDataLayout.BATCH:
      return disw_b
    elif output_dist == FCDataLayout.NEURON:
      return sidw_f
    else:
      assert False
