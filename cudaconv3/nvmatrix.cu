#include "nvmatrix.cuh"
cudaStream_t NVMatrix::_defaultStream = 0;
pthread_mutex_t NVMatrix::_streamMutex;
