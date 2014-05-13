#!/bin/sh
mpirun -np 4 python write.py
mpirun -np 9 python write.py
