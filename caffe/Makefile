SRC = blob.cu caffe_wrap.cu im2col.cu conv_acts.cu common.cu math_functions.cu pool_acts.cu norm_acts.cu
OBJ = $(patsubst %.cu,%.o,${SRC})
CUDA_DIR = /home/justin/pkg/cuda-5.0
CUDA_INCLUDE = -I${CUDA_DIR}/include
CUDA_LIBRARY = -L${CUDA_DIR}/lib

LINK_FLAGS := -lcublas -lcudart -lcurand
NVCC_OPTS := --pre-include undef.h
CXX_FLAGS := --compiler-options='-fPIC -ggdb2' 
PYTHON_INCLUDE := $(shell python-config --includes)
INCLUDES := ${PYTHON_INCLUDE} ${CUDA_INCLUDE} -I.
LIBRARYS := ${CUDA_LIBRARY}

all: _caffe.so

_caffe.so: ${OBJ}
	nvcc -shared -o $@ $^ ${LINK_FLAGS}
 
caffe_wrap.cu: caffe.i Makefile
	swig -python -threads -c++ -o $@ ${INCLUDES} $<

%.o : %.cu Makefile
	nvcc -arch=sm_20 -O3 -g -c -o $@ $<  ${NVCC_OPTS} ${CXX_FLAGS} ${INCLUDES} ${LIBRARYS} ${LINK_FLAGS}

clean:
	rm -f *.o *.so caffe_wrap.cu