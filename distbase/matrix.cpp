#include "matrix.hpp"
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

void trim_images(std::vector<Matrix>& images, Matrix& target, int image_size, int border_size) {
  assert(images.size() == target.getNumRows());
  int inner_size = image_size - 2 * border_size;
  int image_pixel = image_size * image_size;
  int target_pixel = inner_size * inner_size;
  int color = target.getNumCols() / target_pixel;
  for (int i = 0; i < images.size(); i ++ ) {
    int startY = rand() % (border_size * 2 + 1);
    int startX = rand() % (border_size * 2 + 1);
    int flip = rand() % 2;
    flip = 0;
    startY = startX = 0;
    if (flip == 0) {

      for(int c = 0; c < color; c ++) {
        for(int row = 0; row < inner_size; row++) {
          for(int col  = 0; col < inner_size; col ++ ) {
              target.getCell(i, c * target_pixel + row * inner_size + col) = images[i].getCell(c, (row + startY) * image_size + (col + startX));
          }
        }
      }

    } else {

      for(int c = 0; c < color; c ++) {
        for(int row = 0; row < inner_size; row ++) {
          for(int col = 0; col < inner_size; col ++ ) {
              target.getCell(i, c * target_pixel + (inner_size - 1 - row) * inner_size + col) = images[i].getCell(c, (row + startY) * image_size + (col + startX));
          }
        }
      }

    }
  }
}
