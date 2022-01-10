'''
adopted from pytorch.org (Classifying names with a character-level RNN-Sean Robertson)
'''
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import pickle as pkl

import random
import math
import sys
import time
import os
import argparse

from ad_utils import call_model
from ad_model import RNN_enc_RNN_clf, Transformer_enc_RNN_clf
from ad_data import AD_RNN_Dataset
from ad_test import eval_forward, eval_binary, get_valid_loss, log_neptune

from sklearn.metrics import classification_report

def train_main(args, neptune):
    device = torch.device('cuda')

    model = call_model(args, device)

    if args.classifier == 'rnn':
        test_dnn = False
    else:
        test_dnn = True

    train = AD_RNN_Dataset(mode="train",
                            csv_path=args.csv_path,
                            ids_path=args.ids_path,
                            stat_path=args.stat_path,
                            data_name=args.dataset,
                            rnn_len=args.rnn_len,
                            test_dnn=test_dnn)
    valid = AD_RNN_Dataset(mode="valid",
                            csv_path=args.csv_path,
                            ids_path=args.ids_path,
                            stat_path=args.stat_path,
                            data_name=args.dataset,
                            rnn_len=args.rnn_len,
                            test_dnn=test_dnn)
    test = AD_RNN_Dataset(mode="test",
                            csv_path=args.csv_path,
                            ids_path=args.ids_path,
                            stat_path=args.stat_path,
                            data_name=args.dataset,
                            rnn_len=args.rnn_len,
                            test_dnn=test_dnn)

    device = torch.device('cuda')
    criterion = torch.nn.CrossEntropyLoss()

    trainiter = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True)
    validiter = torch.utils.data.DataLoader(valid, batch_size=args.batch_size, shuffle=True)
    testiter = torch.utils.data.DataLoader(test, batch_size=args.batch_size, shuffle=True)
    
    print('trainiter: {} samples'.format(len(train)))
    print('validiter: {} samples'.format(len(valid)))
    print('testiter: {} samples'.format(len(test)))

    # declare optimizer
    estring = "optim." + args.optimizer
    optimizer = eval(estring)(model.parameters(), lr=args.lr)
    if args.use_scheduler == 1:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # modify the dataset to produce labels
    # create a training loop
    train_loss = 0.0
    log_interval = 1000
    log_idx = 0
    bc = 0 # bad counter
    sc = 0 # step counter
    best_valid_loss = None

    for ei in range(args.max_epoch):
        for li, (x_data, y_data) in enumerate(trainiter):
            Bn, Tx, V, D = x_data.size()
            x_data = x_data.to(dtype=torch.float32, device=device) # Bn Tx V D
            y_data = y_data.to(dtype=torch.int64, device=device) # Bn Tx 1

            if args.use_prev_pred == 1:
                loss = 0.0
                y_prev = torch.zeros(Bn,1).to(dtype=torch.int64, device=device) # Bn 1
                clf_hidden = model.init_clf_hidden(Bn, device)

                use_teacher_forcing = True if random.random() < args.teacher_forcing_ratio else False

                for di in range(Tx):
                    # teacher forcing
                    if use_teacher_forcing: # Bn, V, D
                        x_t = x_data[:,di,:,:]
                        y_prev = y_prev.unsqueeze(1).expand(-1,V,-1).contiguous()

                        input_data = torch.cat((x_t, y_prev), dim=-1).unsqueeze(1)
                        
                        output, clf_hidden = model(input_data, clf_hidden)

                        loss += criterion(output, y_data[:, di])

                        y_prev = y_data[:, di].unsqueeze(-1)
                    # free running
                    else:
                        x_t = x_data[:,di,:,:]
                        y_prev = y_prev.unsqueeze(1).expand(-1,V,-1).contiguous()

                        input_data = torch.cat((x_t, y_prev), dim=-1).unsqueeze(1)
                        
                        output, clf_hidden = model(input_data, clf_hidden)

                        loss += criterion(output, y_data[:, di])

                        topv, topi = output.topk(1)
                        y_prev = topi.detach()  # detach from history as input

            else:
                y_data = y_data[:,-1]
    
                output, _ = model(x_data)

                loss = criterion(output, y_data)

            # go through loss function
            optimizer.zero_grad()
            loss.backward()

            # optimizer
            optimizer.step()

            # collect train_loss
            if args.use_prev_pred:
                train_loss += (loss.item() / Tx)
            else:
                train_loss += loss.item()

        train_loss = train_loss / (li + 1)
        print('epoch: {:d} | train_loss: {:.4f}'.format(ei+1, train_loss))
        if neptune is not None: neptune.log_metric('train_loss', ei+1, train_loss)
        train_loss = 0.0

        valid_loss = get_valid_loss(model, validiter, criterion, args.use_prev_pred, device)
        print('epoch: {:d} | valid_loss: {:.4f}'.format(ei+1, valid_loss))
        if neptune is not None: neptune.log_metric('valid_loss', ei+1, valid_loss)

        if ei == 0 or valid_loss < best_valid_loss:
            save_path = args.save_dir + args.out_file
            torch.save(model, save_path)
            bc = 0
            best_valid_loss = valid_loss
            print('found new best model')
        else:
            bc += 1
            if bc > args.patience:
                if args.use_scheduler == 1:
                    print("learning rate decay..")
                    scheduler.step()
                    bc = 0
                    sc += 1

                    if(sc >= args.n_decay):
                        break
                else:
                    print("early stopping..")
                    break

            print('bad counter == %d' % (bc))

    model = torch.load(save_path)

    targets, preds = eval_forward(model, testiter, args.use_prev_pred, device)
    acc, prec, rec, f1 = eval_binary(targets, preds)

    print('testset | acc: {:.4f} | prec: {:.4f} | rec: {:.4f} | f1: {:.4f} |'.format(acc, prec, rec, f1))
    
    if neptune is not None:
        neptune.set_property('acc', acc)
        neptune.set_property('prec', prec)
        neptune.set_property('rec', rec)
        neptune.set_property('f1', f1)

    return

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # task
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--use_neptune', type=int)
    parser.add_argument('--use_prev_pred', type=int)
    parser.add_argument('--teacher_forcing_ratio', type=float)
    # exp_name
    parser.add_argument('--exp_name', type=str)
    # dataset
    parser.add_argument('--dim_input', type=int)
    parser.add_argument('--rnn_len', type=int)
    parser.add_argument('--csv_path', type=str)
    parser.add_argument('--ids_path', type=str)
    parser.add_argument('--stat_path', type=str)
    parser.add_argument('--dict_path', type=str)
    parser.add_argument('--data_name', type=str)
    # feature mapping
    parser.add_argument('--use_feature_mapping', type=int)
    parser.add_argument('--dim_feature_mapping', type=int)
    # enc
    parser.add_argument('--encoder', type=str)
    parser.add_argument('--nlayer', type=int)
    # dnn-enc
    parser.add_argument('--dim_enc', type=int)
    # rnn-enc
    parser.add_argument('--bidirectional', type=int)
    parser.add_argument('--dim_lstm_hidden', type=int)
    # transformer-enc
    parser.add_argument('--nhead', type=int)
    parser.add_argument('--dim_feedforward', type=int)
    # readout
    parser.add_argument('--reduce', type=str)

    # clf
    parser.add_argument('--classifier', type=str)
    parser.add_argument('--clf_n_lstm_layers', type=int)
    parser.add_argument('--clf_n_fc_layers', type=int)
    parser.add_argument('--clf_dim_lstm_hidden', type=int)
    parser.add_argument('--clf_dim_fc_hidden', type=int)
    parser.add_argument('--clf_dim_output', type=int)

    # training parameter
    parser.add_argument('--optimizer', type=str)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--patience', type=float)
    parser.add_argument('--max_epoch', type=int)
    parser.add_argument('--drop_p', type=float)

    # learning rate decay
    parser.add_argument('--use_scheduler', type=int)
    parser.add_argument('--step_size', type=int)
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--n_decay', type=int, default=3)

    args = parser.parse_args()
    params = vars(args)

    args.save_dir = "./result/"
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    if args.use_neptune == 1:
        import neptune
        neptune.init(project="cjlee/supervised-anomaly-detection")
        experiment = neptune.create_experiment(name=args.exp_name, params=params)
        args.out_file = experiment.id + '.pth'
    else:
        neptune=None
        args.out_file = 'dummy.pth'

    print('parameters:')
    print('='*90)
    print(params)
    print('='*90)

    train_main(args, neptune)
