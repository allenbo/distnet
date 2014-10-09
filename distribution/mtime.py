class MTime(object):
  def __init__(self, fprop, bprop, wprop = 0, update = 0):
    self._fprop = fprop
    self._bprop = bprop
    self._wprop = wprop
    self._update = update
    self._sum = self._fprop + self._bprop + self._wprop + self._update

  def get_value(self):
    return self._sum

  @property
  def fprop(self):
    return self._fprop

  @property
  def bprop(self):
    return self._bprop

  @property
  def wprop(self):
    return self._wprop

  @property
  def update(self):
    return self._update

  @property
  def sum(self):
    return self._sum

  def __gt__(self, other):
    return self.sum > other.sum

  def __lt__(self, other):
    return self.sum < other.sum

  def __eq__(self, other):
    return self.sum == other.sum

  def __str__(self):
    return '%f' % (self._sum)
