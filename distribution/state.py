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

combination_conv =(sisw, sidw, disw_b, disw_i)
combination_fc = (sisw, sidw_f, disw_b)
