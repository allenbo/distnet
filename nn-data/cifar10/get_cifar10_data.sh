#!/bin/bash

# get colomn major cifar data
wget http://www.cs.toronto.edu/~kriz/cifar-10-py-colmajor.tar.gz

# uncompress data
tar -xzf cifar-10-py-colmajor.tar.gz
mv cifar-10-py-colmajor old
cp old/*.meta .
python split.py
