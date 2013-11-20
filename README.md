distnet
=========
A [distributed convolutional neural network](http://yann.lecun.com/exdb/lenet/) framework, building on 
top of the convolutional kernel code from [cuda-convnet](https://code.google.com/p/cuda-convnet/).


**Setup**

```
git clone https://github.com/allenbo/distnet
cd distnet
python setup.py develop [--user]
```

**Usage**

To run a trainer directly:

    python distnet/trainer.py --help
    
Take a look at the scripts in `distnet/scripts` for examples of how to run your own network.


**Requires**

  * [NumPy](http://www.numpy.org/)
  * [CUDA](http://www.nvidia.com/object/cuda_home_new.html)
  * [PyCUDA](http://documen.tician.de/pycuda/)
