import os
import sys
import time
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from vocab import Vocab
from utils import helper
from shutil import copyfile
from draw import draw_curve
from sklearn import metrics
from loader import DataLoader
from trainer import GCNTrainer
from torch.autograd import Variable
from load_w2v import load_pretrained_embedding

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='dataset/Restaurants')
parser.add_argument('--vocab_dir', type=str, default='dataset/Restaurants')
parser.add_argument('--glove_dir', type=str, default='dataset/glove')
parser.add_argument('--emb_dim', type=int, default=300, help='Word embedding dimension.')
parser.add_argument('--post_dim', type=int, default=30, help='Position embedding dimension.')
parser.add_argument('--pos_dim', type=int, default=30, help='Pos embedding dimension.')
parser.add_argument('--hidden_dim', type=int, default=50, help='GCN mem dim.')
parser.add_argument('--num_layers', type=int, default=2, help='Num of GCN layers.')
parser.add_argument('--num_class', type=int, default=3, help='Num of sentiment class.')

parser.add_argument('--input_dropout', type=float, default=0.7, help='Input dropout rate.')
parser.add_argument('--gcn_dropout', type=float, default=0.1, help='GCN layer dropout rate.')
parser.add_argument('--lower', default=True, help='Lowercase all words.')
parser.add_argument('--direct', default=False)
parser.add_argument('--loop', default=True)

parser.add_argument('--bidirect', default=True, help='Do use bi-RNN layer.')
parser.add_argument('--rnn_hidden', type=int, default=50, help='RNN hidden state size.')
parser.add_argument('--rnn_layers', type=int, default=1, help='Number of RNN layers.')
parser.add_argument('--rnn_dropout', type=float, default=0.1, help='RNN dropout rate.')

parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
parser.add_argument('--optim', choices=['sgd', 'adagrad', 'adam', 'adamax'], default='adamax', help='Optimizer: sgd, adagrad, adam or adamax.')
parser.add_argument('--num_epoch', type=int, default=100, help='Number of total training epochs.')
parser.add_argument('--batch_size', type=int, default=32, help='Training batch size.')
parser.add_argument('--log_step', type=int, default=20, help='Print log every k steps.')
parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
parser.add_argument('--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

# set random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(1234)
torch.cuda.manual_seed(args.seed)
helper.print_arguments(args)

# load vocab 
print("Loading vocab...")
token_vocab = Vocab.load_vocab(args.vocab_dir + '/vocab_tok.vocab')    # token
post_vocab = Vocab.load_vocab(args.vocab_dir + '/vocab_post.vocab')    # position
pos_vocab = Vocab.load_vocab(args.vocab_dir + '/vocab_pos.vocab')      # POS
dep_vocab = Vocab.load_vocab(args.vocab_dir + '/vocab_dep.vocab')      # deprel
pol_vocab = Vocab.load_vocab(args.vocab_dir + '/vocab_pol.vocab')      # polarity
vocab = (token_vocab, post_vocab, pos_vocab, dep_vocab, pol_vocab)
print("token_vocab: {}, post_vocab: {}, pos_vocab: {}, dep_vocab: {}, pol_vocab: {}".format(len(token_vocab), len(post_vocab), len(pos_vocab), len(dep_vocab), len(pol_vocab)))
args.tok_size = len(token_vocab)
args.post_size = len(post_vocab)
args.pos_size = len(pos_vocab)

# load pretrained word emb
print("Loading pretrained word emb...")
word_emb = load_pretrained_embedding(glove_dir=args.glove_dir, word_list=token_vocab.itos)
assert len(word_emb) == len(token_vocab)
assert len(word_emb[0]) == args.emb_dim
word_emb = torch.FloatTensor(word_emb)                                 # convert to tensor

# load data
print("Loading data from {} with batch size {}...".format(args.data_dir, args.batch_size))
train_batch = DataLoader(args.data_dir + '/train.json', args.batch_size, args, vocab)
test_batch = DataLoader(args.data_dir + '/test.json', args.batch_size, args, vocab)

# check saved_models director
model_save_dir = args.save_dir
helper.ensure_dir(model_save_dir, verbose=True)
# log
file_logger = helper.FileLogger(model_save_dir + '/' + args.log, header="# epoch\ttrain_loss\ttest_loss\ttrain_acc\ttest_acc\ttest_f1")

# build model
trainer = GCNTrainer(args, emb_matrix=word_emb)

# start training
train_acc_history, train_loss_history, test_loss_history, f1_score_history = [], [], [], [0.]
test_acc_history = [0.]
for epoch in range(1, args.num_epoch+1):
    train_loss, train_acc, train_step = 0., 0., 0
    for i, batch in enumerate(train_batch):
        loss, acc = trainer.update(batch)
        train_loss += loss
        train_acc += acc
        train_step += 1
        if train_step % args.log_step == 0:
            print("train_loss: {}, train_acc: {}".format(train_loss/train_step, train_acc/train_step))

    # eval on test
    print("Evaluating on test set...")
    predictions, labels = [], []
    test_loss, test_acc, test_step = 0., 0., 0
    for i, batch in enumerate(test_batch):
        loss, acc, pred, label, _, _ = trainer.predict(batch)
        test_loss += loss
        test_acc += acc
        predictions += pred
        labels += label
        test_step += 1
    # f1 score
    f1_score = metrics.f1_score(labels, predictions, average='macro')

    print("trian_loss: {}, test_loss: {}, train_acc: {}, test_acc: {}, f1_score: {}".format( \
        train_loss/train_step, test_loss/test_step, \
        train_acc/train_step, test_acc/test_step, \
        f1_score))

    file_logger.log("{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}".format( \
        epoch, train_loss/train_step, test_loss/test_step, \
        train_acc/train_step, test_acc/test_step, \
        f1_score))

    train_acc_history.append(train_acc/train_step)
    train_loss_history.append(train_loss/train_step)
    test_loss_history.append(test_loss/test_step)

    # save
    model_file = model_save_dir + '/checkpoint_epoch_{}.pt'.format(epoch)
    trainer.save(model_file)
    # save best model
    if epoch == 1 or test_acc/test_step > max(test_acc_history):
        copyfile(model_file, model_save_dir + '/best_model.pt')
        print("new best model saved.")
        file_logger.log("new best model saved at epoch {}: {:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}"\
            .format(epoch, train_loss/train_step, test_loss/test_step, \
            train_acc/train_step, test_acc/test_step, \
            f1_score))
    test_acc_history.append(test_acc/test_step)
    f1_score_history.append(f1_score)
    print("")

print("Training ended with {} epochs.".format(epoch))
bt_train_acc = max(train_acc_history)
bt_train_loss = min(train_loss_history)
bt_test_acc = max(test_acc_history)
bt_f1_score = f1_score_history[test_acc_history.index(bt_test_acc)]
bt_test_loss = min(test_loss_history)
print("best train_acc: {}, best train_loss: {}, best test_acc/f1_score: {}/{}, best test_loss: {}".format(bt_train_acc, \
                                                                                                          bt_train_loss, \
                                                                                                          bt_test_acc, \
                                                                                                          bt_f1_score, \
                                                                                                          bt_test_loss))
draw_curve(train_log=train_acc_history, test_log=test_acc_history[1:], curve_type="acc", epoch=args.num_epoch)
draw_curve(train_log=train_loss_history, test_log=test_loss_history, curve_type="loss", epoch=args.num_epoch)
#of = open('tmp.txt','a')
#of.write(str(bt_test_acc)+","+str(bt_f1_score)+'\n')
#of.close()
