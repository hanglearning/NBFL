#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=25,
                        help="number of rounds of training")
    parser.add_argument('--n_clients', type=int, default=20,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=100,
                        help="the number of local epochs: E")
    parser.add_argument('--batch_size', type=int, default=10,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--total_samples', type=int, default=200,
                        help="number of images per class per client have")
    parser.add_argument('--alpha', type=float, default=0.5,
                        help="for Dirichlet distribution")


    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to \
                        use for convolution')
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                        of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions")
    
    # pruning arguments
    parser.add_argument('--prune_percent', type=float, default=15,
                        help='pruning percent')
    parser.add_argument('--prune_start_acc', type=float, default=0.2,
                        help='pruning start acc')
    parser.add_argument('--prune_end_rate', type=float, default=0.1,
                        help='pruning end rate')
    parser.add_argument('--mask_ratio', type=float, default=0.5,
                        help='mask ratio')
    parser.add_argument('--rewind', type=float, default=1,
                        help='reset model to initial weights')
    # other arguments
    # parser.add_argument('--dataset', type=str, default='mnist_extr_noniid', help="name \
    #                     of dataset")
    parser.add_argument('--dataset', help="mnist|cifar10",
                        type=str, default="mnist")
    parser.add_argument('--dataset_mode', type=str,
                        default='non-iid', help='non-iid|iid')
    parser.add_argument('--gpu', default=None, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='adam', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=0,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=30, help='random seed')

    # malicious
    parser.add_argument('--n_malicious', type=int, default=0, help='number of malicious users')
    parser.add_argument('--attack_type', type=int, default=0, help='0 - no attack, 1 - model poisoning attack, 2 - label flipping attack, 3 - lazy attack')


    parser.add_argument('--log_dir', type=str, default="./logs")
    parser.add_argument('--num_workers', type=int, default=0) # for pytorch data loader
    parser.add_argument('--noise_variance', type=int, default=1, help="noise variance level of the injected Gaussian Noise")


    args = parser.parse_args()
    return args
