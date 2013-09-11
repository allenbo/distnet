#!/bin/sh

DATADIR=/ssd/nn-data/imagenet

python fastnet/trainer.py \
  --data_dir /ssd/nn-data/imagenet/ \
  --param_file config/imagenet.cfg \
  --data_provider imagenet \
  --train_range 100-1300 \
  --test_range 1-100 \
  --save_freq 100 \
  --test_freq 100 \
  --adjust_freq 100 \
  --learning_rate 0.1 \
  --batch_size 128 \
  --checkpoint_dir checkpoint/ \
  --trainer normal \
  --num_epoch  50 \
  --test_id test-imagenet \
  $@
