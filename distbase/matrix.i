%module matrix
%{
#include "matrix.hpp"
#include "arrayobject.h"
#include <assert.h>
%}

%typemap(in) Matrix& {
  PyObject* shape = PyObject_GetAttrString($input, "shape");
  long  color, rows, cols, batch_size;
  float* data = (float*)(PyArray_DATA((PyArrayObject*)$input));
  PyArg_ParseTuple(shape, "llll",&color, &rows, &cols, &batch_size);
  $1 = new Matrix(color * rows * cols, batch_size, data);
  Py_DECREF(shape);
}

%typemap(freearg) Matrix& {
  delete ($1);
}

%typemap(in) std::vector<Matrix>& {
  int batch_size = PySequence_Size($input);
  $1 = new std::vector<Matrix>();
  for(int i = 0; i < batch_size; i ++ ) {
    PyArrayObject* mobj = (PyArrayObject*)PySequence_GetItem($input, i);
    PyObject* shape = PyObject_GetAttrString((PyObject*)mobj, "shape");
    long  color, rows, cols;
    float* data = (float*)(PyArray_DATA(mobj));
    PyArg_ParseTuple(shape, "lll", &color, &rows, &cols);
    Matrix m(color, rows * cols, data);
    $1->push_back(m);
    Py_DECREF(mobj);
    Py_DECREF(shape);
  }
}

%typemap(freearg) std::vector<Matrix>& {
  delete ($1);
}

%typemap(in) std::vector<JpegData*>& {
  int batch_size = PyList_GET_SIZE($input);
  $1 = new std::vector<JpegData*>();
  for(int i = 0; i < batch_size; i ++ ) {
    PyObject* pySrc = PyList_GET_ITEM($input, i);
    unsigned char* data = (unsigned char*)PyString_AsString(pySrc);
    unsigned int data_size = PyString_GET_SIZE(pySrc);
    JpegData *jd = new JpegData(data, data_size);
    $1->push_back(jd);
  }
}

%typemap(freearg) std::vector<JpegData*>& {
  for(int i = 0; i < ($1)->size(); i ++ ) {
    delete (*($1))[i];
  }
  delete ($1);
}

%include "matrix.hpp"
