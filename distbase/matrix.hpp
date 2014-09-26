#ifndef __MATRIX_HPP__
#define __MATRIX_HPP__

#include <vector>
#include <iostream>

class Matrix {
  public:
    Matrix(int rows, int cols, float* data)
      :_rows(rows), _cols(cols), _data(data) {
      }
    Matrix() {
      _rows = _cols = 0;
      _data = NULL;
    }
    Matrix(const Matrix& other) {
      _rows = other._rows;
      _cols = other._cols;
      _data = other._data;
    }

    inline float& getCell(int row, int col) {
      return _data[row * _cols + col];
    }
    inline void setCell(int row, int col, float value) {
      _data[row * _cols + col] = value;
    }
    void print() {
      if (_data == NULL) return;
      for(int i = 0; i < _rows * _cols; i ++ ) {
        std::cout << _data[i] << " ";
      }
      std::cout << std::endl;
    }

    inline int getNumCols() const { return _cols;}
    inline int getNumRows() const { return _rows;}

  private:
    int _rows;
    int _cols;
    float* _data;
};


void trim_images(std::vector<Matrix>& images, Matrix& target, int image_size, int border_size);
#endif
