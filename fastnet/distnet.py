from fastnet import util
from fastnet.layer import TRAIN, WeightedLayer
from fastnet.net import FastNet
from fastnet.virtual import virtual_array
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


class DistFastNet(FastNet):
  def __init__(self, learning_rate, image_shape, num_ouput, init_model):
    if 'model_state' in init_model:
      model = init_model['model_state']['layers']
    else:
      model = init_model

    for layer in model:
      if layer.type == 'fc':
        layer['outputSize'] = layer['outputSize'] / comm.Get_size()
        size = comm.Get_size()
        if 'weight' in layer:
          layer['weight'] = np.vsplit(layer['weight'], size)[rank]
          layer['bias'] = np.vsplit(layer['bias'], size)[rank]
        if 'weightIncr' in layer:
          layer['weightIncr'] = np.vsplit(layer['weightIncr'], size)[rank]
          layer['biasIncr'] = np.vsplit(layers['biasIncr'], size)[rank]

    FastNet.__init__(self, learning_rate, image_shape, num_ouput, init_model)

  def _log(self, fmt, *args):
    util.log('%s :: %s', rank, fmt % args)


  def fprop(self, data, probs, train = TRAIN):
    fc = False
    input = data
    local_missing = True if not self.local_grads else False
    for i in range(len(self.layers)):
      l = self.layers[i]

      if l.type == 'fc' or l.type == 'softmax':
        fc = True
        input.gather()
        shape = input.get_global_shape()
        if len(shape) == 4:
          c, h, w, b = shape
          input.reshape(c * h * w, b)
        input.distribute(axis = -1)

      if fc != True:
        padding = l.get_cross_width()
        if padding != 0:
          d, area = input.get_cross(padding)
        else:
          d = input.get_local()
          area = input.get_local_area()
        d = d.reshape((d.shape[0] * d.shape[1] * d.shape[2], d.shape[3]))
      else:
        d = input.get_local()
        area = input.get_local_area()

      if local_missing:
        self.local_grads.append(gpuarray.zeros(d.shape, dtype = np.float32))
        self.grads.append(virtual_array(area = area))

      if l.type == 'softmax':
        output = self.output
      else:
        output = self.local_outputs[i]
        input = self.outputs[i]

      l.fprop(d, output, train)
      if not fc:
        shape = self.outputs[i].get_local_shape()
        self.outputs[i].store(output.reshape(shape), self.get_local_area())
      else:
        self.outputs[i].store(output, self.get_local_area())


  def bprop(self, data, label, prob, train = TRAIN):
    grad = label
    fc = True
    for i in range(1, len(self.layers) + 1):
      l = self.layers[-i]
      if l.disableBprop:
        return
      if i == len(self.layers):
        input = data
      else:
        input = self.outputs[-(i+1)]

      if l.type in ['pool', 'rnorm', 'cmrnorm', 'conv']:
        fc = False

      if fc != True:
        padding = l.get_cross_width()
        if padding != 0:
          d, area = input.get_cross(padding)
        else:
          d = input.get_local()
        local_input = d.reshape((d.shape[0] * d.shape[1] * d.shape[2], d.shape[3]))
      else:
        local_input = input.get_local()

      grad.reduce_add()
      if l.type == 'neuron' and fc:
        grad.distribute(axis = 0)
        grad = grad.get_local()
      elif not fc:
        grad.distribute_squre()
        grad = grad.get_local()
        grad = grad.reshape((grad.shape[0] * grad.shape[1] * grad.shape[2] , grad.shape[3]))


      if l.type == 'softmax':
        area = make_plain_area(self.output.shape)
        output = virtual_array(local = self.output, area = area)
        output.gather()
        output.distribute(axis = 0)
        local_output = output.get_local()
      else:
        local_output = self.local_outputs[-i];

      local_outGrad = self.local_grads[-i]
      l.bprop(grad, local_input, local_output, local_outGrad)

      grad = self.grads[-i]
      grad.store(local_outGrad, grad.get_local_area())


  def update(self):
    for layer in self.layers:
      if layer.disableBprop or not isinstance(layer, WeightedLayer):
        continue
      if layer.type == 'fc':
        layer.update()
      else:
        weightGrad, biasGrad = self.weightGrad, self.biasGrad
        area = make_plain_area(weightGrad)
        weightGrad = virtual_array(local = weightGrad, area = area)
        weightGrad.add_reduce()
        weightGrad = wegihGrad.get_local()

        area = make_plain_area(biasGrad)
        biasGrad = virtual_array(local = biasGrad, area = area)
        biasGrad.add_reduce()
        biasGrad = biasGrad.get_local()

        layer.update(weightGrad, biasGrad)

  def prepare_for_train(data, label):
    assert len(data.shape) == 4
    if data.shape[3] != self.batchSize:
      self.batchSize = data.shape[3]
      for l in self.layers:
        l.change_batch_size(self.batchSize)
      self.inputShapes = None
      self.imgShapes = None
      self.outputs = []
      self.grads = []
      self.local_outputs = []
      self.local_grads = []


      self.imgShapes = [(self.numColor, self.imgSize / 2, self.imgSize / 2, self.batchSize)]
      self.inputShapes = [(self.numColr * (self.imgSize ** 2) / 4, self.batchSize)]

      fc = False
      for layer in self.layers:
        outputShape = layer.get_output_shape()

        row = outputShape[0] * outputShape[1] * outputShape[2]
        col = outputShape[3]

        if layer.type == 'softmax':
          row *= comm.Get_size()
          outputShape = (outputShape[0] * comm.Get_size(), 1, 1, outputShape[3])

        self.inputShapes.append((row, col))
        self.imgShapes.append(outputShape)

        area = make_area(outputShape)
        self.outputs.append(virtual_array(rank, area = area))
        self.local_outputs.append(gpuarray.zeros((row, col), dtype =np.float32))

        inputShape = self.inputShapes[-2]
        #if layer.type == 'fc':
        #  inputShape = (inputShape[0] * comm.Get_size(), inputShape[1])
        #  self.local_grads.append(gpuarray.zeors(inputShape, dtype = np.float32))
        #  area = make_plain_area(inputShape)
        #else:
        #  self.local_grads.append(gpuarray.zeros(inputShape, dtype= np.float32))
        #  area = make_area(self.imgShapes[-2])
        #self.grads.append(virtual_array(rank, area = area))

      area = make_area((self.numColor, self.imgSize / 2, self.imgSize / 2, self.batchSize))
      self.data = virtual_array(rank, local = gpuarray.to_gpu(data.__getitem__(area.to_slice())),
          area = area)

      if not isinstance(label, GPUArray):
        self.label = gpuarray.to_gpu(label).astype(np.float32)
      else:
        self.label = label

      self.label = self.label.reshape((label.size, 1))
      self.numCase += data.shape[1]
      outputShape = self.inputShapes[-1]

      if self.output is None or self.output.shape != outputShape:
        self.output = gpuarray.zeros(outputShape, dtype = np.float32)



  def train_batch(self, data, label, train = TRAIN):
    self.prepare_for_train(data, label)
    self.fprop(self.data, self.output, train)
    cost, correct = self.get_cost(self.label, self.output)
    self.cost += cost
    self.correct += correct

    if train == TRAIN:
      self.bprop(self.data, self.label, self.output)
      self.update()

  def get_dumped_layers(self):

    def concatenate(dic, key):
      if key in dic:
        tmp = comm.gather(dic[key])
        comm.barrier()
        dic[key] = np.concatenate(tmp, axis = 0)

    layer_params = []
    for layer in self.layers:
      dic = layer.dump()
      if layer.type == 'fc':
        for key in ['weight', 'bias', 'weightIncr', 'biasIncr']:
          concatenate(dic, key)
      layer_params.append(dic)
      return layer_params




def make_area(shape):
  assert len(shape) == 4
  channel, height, width, batch_size = shape
  if height == 1:
    #input for fc layer
    size = comm.Get_size()
    row_from, row_to = rank * channel / size, (rank + 1) * channel / size - 1
    f = Point(row_from, 0)
    t = Point(row_to, batch_size - 1)
    area = Area(f, t)
  else:
    row_from, row_to, col_from, col_to = rank / 2 * height, rank / 2 * height + height -1, \
            rank % 2 * width, rank % 2 * width + width - 1
    f = Point(0, row_from, col_from, 0)
    t = Point(channel - 1, row_to, col_to, batch_size - 1)
    area = Area(f, t)
  return area

def make_plain_area(shape):
  assert len(shape) == 2
  height, width = shape
  return Area(Point(0, 0), Point(height-1, width-1))
