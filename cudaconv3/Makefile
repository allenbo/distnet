SRC = conv_util.cu cudaconv3_wrap.cu filter_acts.cu img_acts.cu weight_acts.cu nvmatrix.cu
OBJ = $(patsubst %.cu,%.o,${SRC})

NVCC_OPTS := --pre-include undef.h
CXX_FLAGS := --compiler-options='-fPIC'
PYTHON_INCLUDE := $(shell python-config --includes)
#CUDA_SDK_INC := -I/usr/local/cuda/samples/common/inc
CUDA_SDK_INC := -I/usr/local/cuda/samples/common/inc -I./cutil/
INCLUDES := ${PYTHON_INCLUDE} -I. ${CUDA_SDK_INC}

all: _cudaconv3.so

_cudaconv3.so: ${OBJ}
	nvcc -shared -o $@ $^

cudaconv3_wrap.cu: cudaconv3.i Makefile
	swig -python -threads -c++ -o $@ ${INCLUDES} $<

%.o : %.cu Makefile
	nvcc -gencode arch=compute_35,code=sm_35 -m64 -O3 -c -o $@ $<  ${NVCC_OPTS} ${CXX_FLAGS} ${INCLUDES}

clean:
	rm -f *.o *.so
