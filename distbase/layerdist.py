import state

class LayerDist(object):
  def __init__(self, global_dist, group_state, workers_group):
    self._workers_group = worker_groups
    self._num_group = len(self._workers_group)
    self._global_dist = global_dist
    self._group_state = group_state

  @property
  def num_group(self):
    return self._num_group

  @property
  def workers_group(self):
    return self._worker_groups

  @property
  def group_size(self):
    return self._worker_groups[0]

  @property
  def global_dist(self):
    return self._global_dist

  @property
  def group_state(self):
    return self._group_state
