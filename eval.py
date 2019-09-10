import os
import sys
import time
import random
import argparse
import numpy as np
from vocab import Vocab
from shutil import copyfile
from sklearn import metrics
from loader import DataLoader
from trainer import GCNTrainer
from collections import Counter
from utils import torch_utils, helper

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='dataset/Tweets')
parser.add_argument('--vocab_dir', type=str, default='dataset/Tweets')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
parser.add_argument('--lower', default=True, help='Lowercase all words.')
parser.add_argument('--model_dir', type=str, default='saved_models/best_model.pt', help='Directory of the model.')
args = parser.parse_args()

print("Loading model from {}".format(args.model_dir))
opt = torch_utils.load_config(args.model_dir)
loaded_model = GCNTrainer(opt)
loaded_model.load(args.model_dir)

print("Loading vocab...")
token_vocab = Vocab.load_vocab(args.vocab_dir + '/vocab_tok.vocab')    # token
post_vocab = Vocab.load_vocab(args.vocab_dir + '/vocab_post.vocab')    # position
pos_vocab = Vocab.load_vocab(args.vocab_dir + '/vocab_pos.vocab')      # POS
dep_vocab = Vocab.load_vocab(args.vocab_dir + '/vocab_dep.vocab')      # deprel
pol_vocab = Vocab.load_vocab(args.vocab_dir + '/vocab_pol.vocab')      # polarity
vocab = (token_vocab, post_vocab, pos_vocab, dep_vocab, pol_vocab)
print("token_vocab: {}, post_vocab: {}, pos_vocab: {}, dep_vocab: {}, pol_vocab: {}".format(len(token_vocab), len(post_vocab), len(pos_vocab), len(dep_vocab), len(pol_vocab)))

print("Loading data from {} with batch size {}...".format(args.data_dir, args.batch_size))
test_batch = DataLoader(args.data_dir + '/test.json', args.batch_size, args, vocab)
unpacked = helper.unpack_raw_data(test_batch.raw_data, args.batch_size)

print("Evaluating...")
predictions, labels = [], []
test_loss, test_acc, test_step = 0., 0., 0
for i, batch in enumerate(test_batch):
    loss, acc, pred, label, _, _ = loaded_model.predict(batch)
    test_loss += loss
    test_acc += acc
    predictions += pred
    labels += label
    test_step += 1
f1_score = metrics.f1_score(labels, predictions, average='macro')

print("test_loss: {}, test_acc: {}, f1_score: {}".format( \
                                      test_loss/test_step, \
                                      test_acc/test_step, \
                                      f1_score))
