import numpy as np
import argparse
import math
import pickle as pkl
import pandas as pd
import os
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--rnn_len', type=int, default=16)
    parser.add_argument('--train_ratio', type=float)
    parser.add_argument('--valid_ratio', type=float)
    parser.add_argument('--test_ratio', type=float)
    args = parser.parse_args()

    prob = args.train_ratio + args.valid_ratio + args.test_ratio
    if prob != 1.0:
        print("train, valid, and test ratios must equal 1.0")
        sys.exit(-1)

    # load data for number of data
    data = pd.read_csv(args.data_dir + args.dataset + '.csv')

    # obtain indices and ratios
    ids = list(range(len(data)))[args.rnn_len:]
    n_samples = len(ids)

    # split indices and make dicts
    tr_idx = math.ceil(args.train_ratio * n_samples)
    val_idx = math.ceil((args.train_ratio + args.valid_ratio) * n_samples)

    # shuffle
    np.random.shuffle(ids)
    
    tr_ids = ids[:tr_idx]
    val_ids = ids[tr_idx:val_idx]
    test_ids = ids[val_idx:]

    out_dict = {'train':tr_ids, 'valid':val_ids, 'test':test_ids}
    target_path = args.data_dir + args.dataset + ".indices.rnn_len" + str(args.rnn_len) + ".pkl"

    with open(target_path, 'wb') as fp:
        pkl.dump(out_dict, fp)