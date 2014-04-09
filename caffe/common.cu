// Copyright 2013 Yangqing Jia

#include <cstdio>
#include <ctime>
#include <sys/types.h>
#include <unistd.h>

#include "common.cuh"

long cluster_seedgen(void) {
  long s, seed, pid;
  pid = getpid();
  s = time(NULL);
  seed = abs(((s * 181) * ((pid - 83) * 359)) % 104729);
  return seed;
}
