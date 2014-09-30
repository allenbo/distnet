#ifndef __MATRIX_HPP__
#define __MATRIX_HPP__

#include <vector>
#include <iostream>
#include <stdlib.h>
#include <pthread.h>
#include <jpeglib.h>
#include <assert.h>
#include <stdio.h>

class JpegData {
  public:
    JpegData(unsigned char* data, unsigned int data_size)
      :_data(data), _data_size(data_size) {
    }
    JpegData()
      :_data(NULL) {
      _data_size = 0;
    }
    JpegData(const JpegData& other)
      :_data(other._data){
      _data_size = other._data_size;
    }
    inline const unsigned char* data() const { return _data;}
    inline unsigned int data_size() const { return _data_size;}

  private:
    const unsigned char* _data;
    unsigned int _data_size;
};

class Matrix {
  public:
    Matrix(int rows, int cols, float* data)
      :_rows(rows), _cols(cols), _data(data) {
        _own_data = false;
      }
    Matrix() {
      _rows = _cols = 0;
      _data = NULL;
      _own_data = false;
    }
    Matrix(int rows, int cols) {
      _rows = rows;
      _cols = cols;
      _data = new float[_rows * _cols];
      _own_data = true;
    }

    Matrix(const Matrix& other) {
      _rows = other._rows;
      _cols = other._cols;
      _data = other._data;
      _own_data = false;
    }
    ~Matrix() {
      if (_own_data && _data) {
        delete [] _data;
      }
    }

    inline float& getCell(int row, int col) {
      return _data[row * _cols + col];
    }

    inline int getNumCols() const { return _cols;}
    inline int getNumRows() const { return _rows;}

  private:
    int _rows;
    int _cols;
    float* _data;
    bool _own_data;
};

class TrimThread;
class DecodeThread;

class TrimThread {
  public:
    TrimThread(std::vector<Matrix>& images, Matrix& target, int image_size, int border_size, int start, int end)
      :_images(images), _target(target), _image_size(image_size), _border_size(border_size), _start(start), _end(end) {
    }
    static void* run(void* context) {
      TrimThread* self = (TrimThread*)context;
      int inner_size = self->_image_size - 2 * self->_border_size;
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



class DecodeTrimThread {
  public:
    DecodeTrimThread(std::vector<JpegData*>& jpegs, Matrix& target, int image_size, int border_size, int start, int end)
      :_jpegs(jpegs), _target(target), _image_size(image_size), _border_size(border_size) , _start(start), _end(end){
        _inner_size = _image_size - 2 * _border_size;
        _target_pixel = _inner_size * _inner_size;
        _image_pixel = _image_size * _image_size;
        _color = _target.getNumRows() / _target_pixel;
        _decoded_data = NULL;
        _decoded_size = 0;
    }
    ~DecodeTrimThread() {
      if (_decoded_data != NULL) {
        free(_decoded_data);
      }
    }

    static void* run(void* context) {
      DecodeTrimThread* self = (DecodeTrimThread*)context;
      for(int i = self->_start; i < self->_end; i ++ ) {
        self->decode(i);
        self->crop(i);
      }
    }

    void decode(int i) {
      const unsigned char* data = _jpegs[i]->data();
      unsigned int data_size = _jpegs[i]->data_size();

      struct jpeg_decompress_struct cinf;
      struct jpeg_error_mgr jerr;

      cinf.err = jpeg_std_error(&jerr);
      jpeg_create_decompress(&cinf);

      jpeg_mem_src(&cinf, (unsigned char*)data, data_size);

      assert(jpeg_read_header(&cinf, TRUE));

      cinf.out_color_space = JCS_RGB;
      assert(jpeg_start_decompress(&cinf));
      assert(cinf.num_components == 3 || cinf.num_components == 1);

      int width = cinf.image_width;
      int height = cinf.image_height;
      assert(width == _image_size && height == _image_size);

      if (_decoded_size < width * height * 3) {
          if(_decoded_data != NULL)
            free(_decoded_data);
          _decoded_size = width * height * 3 * 3;
          _decoded_data = (unsigned char*)malloc(_decoded_size);
      }

      while (cinf.output_scanline < cinf.output_height) {
          JSAMPROW tmp = &_decoded_data[width * cinf.out_color_components * cinf.output_scanline];
          assert(jpeg_read_scanlines(&cinf, &tmp, 1) > 0);
      }
      assert(jpeg_finish_decompress(&cinf));
      jpeg_destroy_decompress(&cinf);
    }

    void crop(int i) {
      int startY = rand() % (_border_size * 2 + 1);
      int startX = rand() % (_border_size * 2 + 1);
      int flip = rand() % 2;
      if (flip == 0) {

        for(int c = 0; c < _color; c ++) {
          for(int row = 0; row < _inner_size; row++) {
            for(int col  = 0; col < _inner_size; col ++ ) {
                _target.getCell(c * _target_pixel + row * _inner_size + col, i) = _decoded_data[3 *((row + startY) * _image_size + (col + startX)) + c];
            }
          }
        }

      } else {

        for(int c = 0; c < _color; c ++) {
          for(int row = 0; row < _inner_size; row ++) {
            for(int col = 0; col < _inner_size; col ++ ) {
                _target.getCell(c * _target_pixel + (_inner_size - 1 - row) * _inner_size + col, i) = _decoded_data[3 *((row + startY) * _image_size + (col + startX)) + c];
            }
          }
        }
      }

    }
  private:
    std::vector<JpegData*>& _jpegs;
    Matrix& _target;
    unsigned char* _decoded_data;
    unsigned int _decoded_size;
    int _image_size;
    int _border_size;
    int _inner_size;
    int _target_pixel;
    int _image_pixel;
    int _color;
    int _start;
    int _end;

};

void trim_images(std::vector<Matrix>& images, Matrix& target, int image_size, int border_size);
void decode_trim_images(std::vector<JpegData*>& jpegs, Matrix& target, int iamge_size, int border_size);
#endif
