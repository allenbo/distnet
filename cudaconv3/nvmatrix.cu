#include "nvmatrix.cuh"
std::map<int, cudaStream_t> NVMatrix::_defaultStreams;
pthread_mutex_t NVMatrix::_streamMutex;
