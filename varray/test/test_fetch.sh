#!/bin/sh
mpirun -np 4 python fetch.py
mpirun -np 9 python fetch.py
