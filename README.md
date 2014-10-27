DistNet
=========
A [distributed convolutional neural network](http://yann.lecun.com/exdb/lenet/) framework, building on
top of the convolutional kernel code from [cuda-convnet](https://code.google.com/p/cuda-convnet/)


**Requires**

  * MPI:[OpenMPI](http://www.open-mpi.org) or [MVAPICH](http://mvapich.cse.ohio-state.edu/)
  * [CUDA](http://www.nvidia.com/object/cuda_home_new.html)
  * [JPEGLIB](http://www.ijg.org/)

  * [NumPy](http://www.numpy.org/)
  * [PyCUDA](http://documen.tician.de/pycuda/)
  * [mpi4py](http://mpi4py.scipy.org/)
  * [PIL](http://www.pythonware.com/products/pil/)
  * [scikits-cuda](http://scikits.appspot.com/cuda)
  * [yappi](https://code.google.com/p/yappi/)
  * [Cython](http://cython.org/)

If your GPU cluster is hosted in a single machine, we highly recommend you install [MVAPICH](http://mvapich.cse.ohio-state.edu)
as MPI implementation in order to maintain a more stable GPU memory access. You should enable CUDA support when compiling any MPI
implementation. If your GPU cluster is hosted in multiply machines connected by infiniband, please enable infiniband support
when compiling MPI. Please set your `mpirun` binary in your `PATH`, to check you have MPI installed correctly, run `mpirun -np 4 echo "Hello MPi"` in your shell
and make sure you see 4 "Hello MPI". After MPI implementation has installed, you should be able to install `mpi4py`.

You have to install CUDA compiler and SDK before you install PyCuda. Please download CUDA from [CUDA](https://developer.nvidia.com/cuda-downloads) and follow instructions
to install it. After you install CUDA, please put your NVCC compiler binary in your `PATH` variable for installing PyCuda.

**Setup**

```
git clone https://github.com/allenbo/distnet
cd distnet
python setup.py develop [--user]
```

After installing all packages in dependancy, you should be able to install distnet successfully.

**Usage**

This is a demo to train a cifar10 images.

  * To get cifar10 and preprocess data:

	```
	cd nn-data/cifar10 && ./get_cifar10_data.sh
  	```
  * To train cifar10 directly:

	```
	python experiments/run_cifar10_demo.py
	```

  * To train cifar10 in MultiGPU:

	```
	For OpenMPI: mpirun -np 4 [--hostfile hostfile] -x MULTIGPU=yes python experiments/run_cifar10_demo.py
	For MVAPICH: mpirun -np 4 [--hostfile hostfile] -env MULTIGPU=yes -env MV2_USE_CUDA=1 python experiments/run_cifar10_demo.py
	```
If your GPUs are in different host machines, you need to provide a hostfile for mpirun.

In order to train ImageNet, you have to download ImageNet train dataset. We preprocess ImageNet train dataset and calculate the mean of the entire dataset and save it to nn-data/imagenet/batches.meta. If you have ImageNet dataset installed in your machine, please move the mean data file into your data directory and change the `data_dir` variable in `experiments/run_imagenet_simple.py`.

You can use the same command to train the ImageNet dataset.

```
python experiments/run_imagenet_simple.py
```

Take a look at the scripts in `distnet/experiments` for examples of how to run your own network.


