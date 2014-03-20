%module caffe
%{
#include "common.cuh"
#include "blob.cuh"
#include "syncedmem.cuh"
#include "caffe.cuh"
%}

%typemap(in) Blob& {
  PyObject* shape = PyObject_GetAttrString($input, "shape");
  long channel = 1, rows, cols, batch_size = 1;
  PyObject* data = PyObject_GetAttrString($input, "gpudata");
  Py_DECREF(shape);
  PyObject* strides = PyObject_GetAttrString($input, "strides");
  Py_DECREF(strides);

  long stride, itemsize;

  float* gpudata = (float*)PyInt_AsLong(data);
  Py_DECREF(data);
  int len = PyTuple_Size(shape);
  if (len == 2){
    PyArg_ParseTuple(shape, "ll", &rows, &cols);
    //printf("The shape is (%ld, %ld)\n", rows, cols);
    //printf("The stride is %ld\n", stride);
  }else if (len == 4) {
    PyArg_ParseTuple(shape, "llll", &channel, &rows, &cols, &batch_size);
    //printf("The size of shape is 4, and the shape is (%ld, %ld, %ld, %ld)\n", channel, rows, cols, batch_size);
    //printf("Shape changes to (%ld, %ld)\n", rows, cols);
    long stride0, stride1, stride2, itemsize;
    PyArg_ParseTuple(strides, "llll", &stride0, &stride1, &stride2, &itemsize);
    //printf("The strides is (%ld, %ld, %ld, %ld)\n", stride0, stride1, stride2, itemsize);
    stride = stride2/itemsize;
    //printf("Strides change to %ld\n", stride);
  }
  $1 = new Blob(batch_size, channel, rows, cols, batch_size* channel* rows* cols, gpudata);
}

%include "caffe.cuh"

%typemap(typecheck,precedence=SWIG_TYPECHECK_INTEGER) Blob& {
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



%inline %{
PyObject* make_buffer(long offset, long size) {
  return PyBuffer_FromReadWriteMemory((void*)offset, size);
}

%}
