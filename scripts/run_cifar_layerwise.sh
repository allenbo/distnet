#!/bin/sh


dir=~/fastnet
python $dir/fastnet/trainer.py \
  --data_dir /ssd/nn-data/cifar-10.old/ \
  --param_file $dir/config/cifar_from_imagenet.cfg \
  --data_provider cifar10 \
  --train_range 1-40 \
  --test_range 41-48 \
  --save_freq 100 \
  --test_freq 100 \
  --adjust_freq 100 \
  --learning_rate 1.0 \
  --batch_size 128 \
  --checkpoint_dir $dir/fastnet/checkpoint/ \
  --trainer layerwise \
  --output_dir /scratch1/justin/cifar-pickle/ \
  --trainer layerwise \
  --replaynet_epoch 1 \
  --frag_epoch 1 \
  --num_epoch 30 \
  $@
