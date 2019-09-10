#!/bin/bash
# build vocab for different datasets
#python prepare_vocab.py --data_dir dataset/Restaurants --vocab_dir dataset/Restaurants
#python prepare_vocab.py --data_dir dataset/Laptops --vocab_dir dataset/Laptops
#python prepare_vocab.py --data_dir dataset/Restaurants16 --vocab_dir dataset/Restaurants16
python prepare_vocab.py --data_dir dataset/Tweets --vocab_dir dataset/Tweets
