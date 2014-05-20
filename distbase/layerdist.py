import state

class LayerDist(object):
  def __init__(self, global_dist, group_state, workers_group):
    self._workers_group = workers_group
    self._num_group = len(self._workers_group)
    self._global_dist = global_dist
    self._group_state = group_state

  @property
  def num_group(self):
    return self._num_group

  @property
  def workers_group(self):
    return self._workers_group

  @property
  def group_size(self):
    return self._workers_group[0]

  @property
  def global_dist(self):
    return self._global_dist

  @property
  def group_state(self):
    return self._group_state

  def __str__(self):
    return '[Layer Distribution]GlobalDist:%s, GroupState:%s, Workers:%s' % (self._global_dist, self._group_state, self._workers_group)
