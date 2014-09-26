#ifndef __MATRIX_HPP__
#define __MATRIX_HPP__

#include <vector>
#include <iostream>
#include <stdlib.h>
#include <pthread.h>

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

class TrimThread;
class TrimThread {
  public:
    TrimThread(std::vector<Matrix>& images, Matrix& target, int image_size, int border_size, int start, int end)
      :_images(images), _target(target), _image_size(image_size), _border_size(border_size), _start(start), _end(end) {
    }
    static void* run(void* context) {
      TrimThread* self = (TrimThread*)context;
      int inner_size = self->_image_size - self->_border_size;
      int target_pixel = inner_size * inner_size;
      int image_pixel = self->_image_size * self->_image_size;
      int color = self->_target.getNumRows() / target_pixel;

      for (int i = self->_start; i < self->_end; i ++ ) {
        int startY = rand() % (self->_border_size * 2 + 1);
        int startX = rand() % (self->_border_size * 2 + 1);
        int flip = rand() % 2;
        if (flip == 0) {

          for(int c = 0; c < color; c ++) {
            for(int row = 0; row < inner_size; row++) {
              for(int col  = 0; col < inner_size; col ++ ) {
                  self->_target.getCell(c * target_pixel + row * inner_size + col, i) = self->_images[i].getCell(c, (row + startY) * self->_image_size + (col + startX));
              }
            }
          }

        } else {

          for(int c = 0; c < color; c ++) {
            for(int row = 0; row < inner_size; row ++) {
              for(int col = 0; col < inner_size; col ++ ) {
                  self->_target.getCell(c * target_pixel + (inner_size - 1 - row) * inner_size + col, i) = self->_images[i].getCell(c, (row + startY) * self->_image_size + (col + startX));
              }
            }
          }

        }
      }
    }
  private:
    std::vector<Matrix>& _images;
    Matrix& _target;
    int _image_size;
    int _border_size;
    int _start;
    int _end;
};

void trim_images(std::vector<Matrix>& images, Matrix& target, int image_size, int border_size);
#endif
