#!/bin/bash

python experiments/run_cifar10_simple.py > single
num=`wc -l single | cut -d" " -f 1`
echo ${num}
for i in `seq 100`; do
  mpirun -np 4 -x MULTIGPU=yes python experiments/run_cifar10_simple.py > multi
  head -n ${num} multi > cropped
  X=`diff cropped single`
  [[ "${X}" != "" ]] && exit 1
  echo ${i}
  mv multi multi${i}
  pkill -f python
  pkill -f mpirun
  sleep 1
  
done
