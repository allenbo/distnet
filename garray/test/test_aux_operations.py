import garray
import numpy as np

garray.device_init()

# only test out relu_activate since other operations were already tested out at distbase

shape = (128, 96, 55, 55)
input_local = np.random.randn(*shape).astype(np.float32)

input = garray.array(input_local)

garray.relu_activate(input, input, 0)
input_local[input_local < 0] = 0

assert (input_local == input.get()).all()
