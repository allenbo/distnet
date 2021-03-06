%module cudaconv3
%{
#include "cudaconv2.cuh"
#include "conv_util.cuh"
#include "nvmatrix.cuh"
%}

%typemap(in) NVMatrix& {
  PyObject* shape = PyObject_GetAttrString($input, "shape");
  long channel, rows, cols, batch_size;
  PyObject* data = PyObject_GetAttrString($input, "gpudata");
  Py_DECREF(shape);
  PyObject* strides = PyObject_GetAttrString($input, "strides");
  Py_DECREF(strides);
  PyObject* stream = PyObject_GetAttrString($input, "stream");
  Py_DECREF(stream);
  PyObject* stream_handle = PyObject_GetAttrString(stream, "handle");

  long stride, itemsize;
  int stream_int = (int)PyInt_AsLong(stream_handle);

  float* gpudata = (float*)PyInt_AsLong(data);
  Py_DECREF(data);
  int len = PyTuple_Size(shape);
  if (len == 2){
    PyArg_ParseTuple(shape, "ll", &rows, &cols);
    PyArg_ParseTuple(strides, "ll", &stride, &itemsize);
    stride = stride / itemsize;
  }else if (len == 4) {
    PyArg_ParseTuple(shape, "llll", &channel, &rows, &cols, &batch_size);
    rows = channel * rows * cols;
    cols = batch_size;
    long stride0, stride1, stride2, itemsize;
    PyArg_ParseTuple(strides, "llll", &stride0, &stride1, &stride2, &itemsize);
    stride = stride2/itemsize;
  }
  $1 = new NVMatrix(gpudata, rows, cols, stride, (cudaStream_t)stream_int);
}

%typemap(freearg) NVMatrix& {
  delete ($1);
}

%typemap(typecheck,precedence=SWIG_TYPECHECK_INTEGER) NVMatrix& {
  if (PyObject_HasAttrString($input, "shape") && PyObject_HasAttrString($input, "gpudata")) {
    $1 = 1;
  } else {
    $1 = 0;
  }
}

%exception {
  try {
    $function
  } catch (Exception& e) {
    PyErr_Format(PyExc_RuntimeError, "%s (%s:%d", e.why_.c_str(), e.file_.c_str(), e.line_);
    return NULL;
  }
}


%include "cudaconv2.cuh"
%include "nvmatrix.cuh"

void sum(NVMatrix& src, int axis, NVMatrix& target);
void addVector(NVMatrix& target, NVMatrix& vec);

void convLocalMaxPool(NVMatrix& images, NVMatrix& target, int numFilters,
                   int subsX, int startX, int strideX, int imageY, int outputsY, int outputsX);
void convLocalAvgPool(NVMatrix& images, NVMatrix& target, int numFilters,
                   int subsX, int startX, int strideX, int imageY, int outputsY, int outputsX);

void convLocalMaxUndo(NVMatrix& images, NVMatrix& maxGrads, NVMatrix& maxActs, NVMatrix& target,
                      int subsX, int startX, int strideX, int outputsY, int outputsX, int imageY);
void convLocalMaxUndo(NVMatrix& images, NVMatrix& maxGrads, NVMatrix& maxActs, NVMatrix& target,
                      int subsX, int startX, int strideX, int outputsY, int outputsX, int imageY, float scaleTargets, float scaleOutput);

void convLocalAvgUndo(NVMatrix& avgGrads, NVMatrix& target,
                      int subsX, int startX, int strideX, int outputsY, int outputsX, int imageY, int imageX);

void convLocalAvgUndo(NVMatrix& avgGrads, NVMatrix& target,
                      int subsX, int startX, int strideX, int outputsY, int outputsX, int imageY, int imageX,
                      float scaleTargets, float scaleOutput);

void convResponseNorm(NVMatrix& images, NVMatrix& denoms, NVMatrix& target, int numFilters, int sizeX, int imageY, float addScale, float powScale);
void convResponseNormUndo(NVMatrix& outGrads, NVMatrix& denoms, NVMatrix& inputs, NVMatrix& acts, NVMatrix& target, int numFilters,
                         int sizeX, int imageY, float addScale, float powScale, float scaleTargets, float scaleOutput);

void convResponseNormCrossMap(NVMatrix& images, NVMatrix& denoms, NVMatrix& target, int numFilters, int sizeF, int imageY, float addScale,
                              float powScale, bool blocked);
void convResponseNormCrossMapUndo(NVMatrix& outGrads,NVMatrix& denoms, NVMatrix& inputs, NVMatrix& acts, NVMatrix& target, int numFilters,
                         int sizeF, int imageY, float addScale, float powScale, bool blocked, float scaleTargets, float scaleOutput);

%inline %{
PyObject* make_buffer(long offset, long size) {
  return PyBuffer_FromReadWriteMemory((void*)offset, size);
}

%}
