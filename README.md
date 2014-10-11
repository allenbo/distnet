distnet
=========
A [distributed convolutional neural network](http://yann.lecun.com/exdb/lenet/) framework, building on 
top of the convolutional kernel code from [cuda-convnet](https://code.google.com/p/cuda-convnet/) and [caffe](https://github.com/BVLC/caffe/)


**Setup**

```
git clone https://github.com/allenbo/distnet
cd distnet
python setup.py develop [--user]
```

**Usage**

To run cifar-10 directly:

    python experiments/run_cifar10_simple.py
    
Take a look at the scripts in `distnet/experiments` for examples of how to run your own network.


**Requires**

  * [NumPy](http://www.numpy.org/)
  * [CUDA](http://www.nvidia.com/object/cuda_home_new.html)
  * [PyCUDA](http://documen.tician.de/pycuda/)
  * MPI:[OpenMPI](http://www.open-mpi.org) or [MVAPICH](http://http://mvapich.cse.ohio-state.edu/)
  * [mpi4py](http://mpi4py.scipy.org/)
  * [PIL](http://www.pythonware.com/products/pil/)
  * [scikits-cuda](http://scikits.appspot.com/cuda)
  * [yappi](https://code.google.com/p/yappi/)
  * [Cython](http://cython.org/)
