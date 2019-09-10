#!/bin/bash
# training command for different datasets.

python train.py --data_dir dataset/Restaurants --vocab_dir dataset/Restaurants --num_layers 2 --seed 29
#python train.py --data_dir dataset/Laptops --vocab_dir dataset/Laptops --num_layers 2 --seed 6
#python train.py --data_dir dataset/Restaurants16 --vocab_dir dataset/Restaurants16 --num_layers 2 --post_dim 0 --pos_dim 0 --input_dropout 0.8 --num_epoch 200 --seed 0
#python train.py --data_dir dataset/Tweets --vocab_dir dataset/Tweets --num_layers 7 --seed 124
