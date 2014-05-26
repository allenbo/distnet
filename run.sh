#!/bin/bash

TMPDIR=/tmp
EXPERIMENTS=experiments/run_imagenet_dummy.py
MULTI=${TMPDIR}/multi
SINGLE=${TMPDIR}/single
CROPPED=${TMPDIR}/cropped

python  ${EXPERIMENTS} > ${SINGLE}
num=`wc -l ${SINGLE} | cut -d" " -f 1`
echo ${num}
for i in `seq 1`; do
  mpirun -np 4 --hostfile hostfile -x MULTIGPU=yes python ${EXPEIRMENTS} > ${MULTI}
  head -n ${num} ${MULTI} > ${CROPPED}
  X=`diff ${CROPPED} ${SINGLE}`
  [[ "${X}" != "" ]] && exit 1
done
