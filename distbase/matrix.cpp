#include "matrix.hpp"
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

#define NUM_THREAD 16

void trim_images(std::vector<Matrix>& images, Matrix& target, int image_size, int border_size) {
  assert(images.size() == target.getNumCols());
  int num_image_per_thread = images.size() / NUM_THREAD;
  TrimThread* thread_context[NUM_THREAD];
  pthread_t thread_id[NUM_THREAD];
  for(int i = 0; i < NUM_THREAD; i ++ ) {
    thread_context[i] = new TrimThread(images, target, image_size, border_size, i * num_image_per_thread, (i+1) * num_image_per_thread);
    pthread_create(&thread_id[i], NULL, TrimThread::run, (void*)thread_context[i]);
  }

  for(int i = 0; i < NUM_THREAD; i ++ ) {
    pthread_join(thread_id[i], NULL);
    delete thread_context[i];
  }
}

void decode_trim_images(std::vector<JpegData*>& jpegs, Matrix& target, int image_size, int border_size) {
  assert(jpegs.size() == target.getNumCols());
  int num_image_per_thread = jpegs.size() / NUM_THREAD;
  DecodeTrimThread* thread_context[NUM_THREAD];
  pthread_t thread_id[NUM_THREAD];
  for(int i = 0; i < NUM_THREAD; i ++ ) {
    thread_context[i] = new DecodeTrimThread(jpegs, target, image_size, border_size, i * num_image_per_thread, (i+1) * num_image_per_thread);
    pthread_create(&thread_id[i], NULL, DecodeTrimThread::run, (void*)thread_context[i]);
  }

  for(int i = 0; i < NUM_THREAD; i ++ ) {
    pthread_join(thread_id[i], NULL);
    delete thread_context[i];
  }
}
