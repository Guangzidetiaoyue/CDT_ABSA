import os
import random

i = 45
while i <= 500:
    os.system('python train.py --data_dir dataset/Tweets --vocab_dir dataset/Tweets --num_layers 7 --seed '+str(i))
    i += 1
