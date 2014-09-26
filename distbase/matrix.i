%module matrix
%{
#include "matrix.hpp"
#include "arrayobject.h"
%}

%typemap(in) Matrix& {
  PyObject* shape = PyObject_GetAttrString($input, "shape");
  long  color, rows, cols, batch_size;
  float* data = (float*)(PyArray_DATA((PyArrayObject*)$input));
  PyArg_ParseTuple(shape, "llll", &batch_size, &color, &rows, &cols);
  $1 = new Matrix(batch_size, color * rows * cols, data);
  Py_DECREF(shape);
}

%typemap(freearg) Matrix& {
  delete ($1);
}

%typemap(in) std::vector<Matrix>& {
  int batch_size = PySequence_Size($input);
  $1 = new std::vector<Matrix>[batch_size];
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
  delete []($1);
}
%include "matrix.hpp"
