PYTHON_INCLUDE := $(shell python-config --includes)
NUMPY_INCLUDE := -I${HOME}/.local/lib/python2.7/site-packages/numpy/core/include/numpy/
INCLUDE := ${PYTHON_INCLUDE} ${NUMPY_INCLUDE}

all: _matrix.so

_matrix.so: matrix.o matrix_wrap.o
	g++ -shared -O3 -m64 -o $@ $^ -lpthread -ljpeg

matrix_wrap.cpp: matrix.i Makefile
	swig -python -threads -c++ -o $@ ${PYTHON_INCLUDE} $<

%.o:%.cpp Makefile
	g++ -O3 -m64 -c -o $@ $< -fPIC ${INCLUDE}

clean:
	rm -f *.o *.so matrix_wrap.cpp matrix.py*
