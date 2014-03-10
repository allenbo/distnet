import threading, time
import garray
import numpy as np


lock = threading.Lock()


class MyThread(threading.Thread):
  def __init__(self, word):
    self.word = word
    threading.Thread.__init__(self)

  def run(self):
    while( 100000 ):
      time.sleep(0.2)
      with lock:
        print self.word


a = garray.array(np.random.randn(4000, 6000).astype(np.float32))
b = garray.array(np.random.randn(6000, 7000).astype(np.float32))

MyThread("thread").start()

last = time.time()
for i in range(100000):
  c = garray.dot(a, b)
  if time.time() - last > 1:
    with lock:
      print "main"
    last = time.time()
