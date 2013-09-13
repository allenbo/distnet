#!/bin/sh


  #--param_file ~/fastnet/fastnet/fastcifar.cfg \
python ./fastnet/trainer.py \
  --data_dir /ssd/nn-data/cifar-10.old \
  --data_provider cifar10 \
  --train_range 1-40 \
  --test_range 40-48 \
  --save_freq 500 \
  --test_freq 500 \
  --adjust_freq 100 \
  --learning_rate 1.0 \
  --batch_size 128 \
  --param_file=config/cifar-10-18pct.cfg \
  --checkpoint_dir checkpoint/ \
  --trainer normal \
  --num_epoch 30 \
  $@
