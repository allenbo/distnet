SRC = blob.cu im2col.cu filter_acts.cu syncedmem.cu common.cu
OBJ = $(patsubst %.cu,%.o,${SRC})
CUDA_DIR = /home/justin/pkg/cuda-5.0
CUDA_INCLUDE = -I${CUDA_DIR}/include
CUDA_LIBRARY = -L${CUDA_DIR}/lib

NVCC_OPTS := --pre-include undef.h
CXX_FLAGS := --compiler-options='-fPIC -ggdb2' -lcublas
PYTHON_INCLUDE := $(shell python-config --includes)
INCLUDES := ${PYTHON_INCLUDE} ${CUDA_INCLUDE}
LIBRARY := ${CUDA_LIBRARY}

all: _caffe.so

_caffe.so: ${OBJ}
	nvcc -shared -o $@ $^

caffe_wrap.cu: caffe.i Makefile
	swig -python -threads -c++ -o $@ ${INCLUDES} $<

%.o : %.cu Makefile
	nvcc -arch=sm_20 -O3 -g -c -o $@ $<  ${NVCC_OPTS} ${CXX_FLAGS} ${INCLUDES} ${LIBRARY}

clean:
	rm -f *.o *.so
