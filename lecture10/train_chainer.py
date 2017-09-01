#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Takuma Yagi <tyagi@iis.u-tokyo.ac.jp>
#
# Distributed under terms of the MIT license.

import argparse
import time
import os
import pickle

import numpy as np
import chainer
from chainer import Variable, optimizers, serializers, cuda

from utils import DataLoader
from model_chainer import Model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1),
    parser.add_argument('--resume', type=str, default="")
    parser.add_argument('--nb_units', type=int, default=256,
                        help='size of RNN hidden state')
    parser.add_argument('--nb_layers', type=int, default=2,
                        help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='lstm',
                        help='rnn, gru, or lstm')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=300,
                        help='RNN sequence length')
    parser.add_argument('--nb_epochs', type=int, default=30,
                        help='number of epochs')
    parser.add_argument('--save_every', type=int, default=500,
                        help='save frequency')
    parser.add_argument('--model_dir', type=str, default='save',
                        help='directory to save model to')
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients at this value')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.95,
                        help='decay rate for rmsprop')
    parser.add_argument('--nb_mixtures', type=int, default=20,
                        help='number of gaussian mixtures')
    parser.add_argument('--data_scale', type=float, default=20,
                        help='factor to scale raw data down by')
    parser.add_argument('--keep_prob', type=float, default=0.8,
                        help='dropout keep probability')
    args = parser.parse_args()

    data_loader = DataLoader(args.batch_size, args.seq_length, args.data_scale)

    if args.model_dir != '' and not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    with open(os.path.join(args.model_dir, 'config.pkl'), 'wb') as f:
        pickle.dump(args, f)

    model = Model(args.nb_layers, args.nb_units,
                  args.nb_mixtures, args.data_scale,
                  args.keep_prob)

    if args.resume != "":
        serializers.load_npz(args.resume, model)

    if args.gpu >= 0:
        model.to_gpu(args.gpu)

    optimizer = optimizers.Adam(alpha=0.0)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.grad_clip))

    chainer.config.use_cudnn = "never"

    for e in range(args.nb_epochs):
        optimizer.alpha = args.lr * (args.decay_rate ** e)
        data_loader.reset_batch_pointer()
        v_x, v_y = data_loader.validation_data()
        v_x = np.array(v_x)
        v_y = np.array(v_y)

        if args.gpu >= 0:
            v_x = cuda.to_gpu(v_x, args.gpu)
            v_y = cuda.to_gpu(v_y, args.gpu)

        sum_loss = 0
        for b in range(data_loader.num_batches):
            i = e * data_loader.num_batches + b
            start = time.time()
            x, y = data_loader.next_batch()
            x = np.array(x)
            y = np.array(y)

            if args.gpu >= 0:
                x = cuda.to_gpu(x, args.gpu)
                y = cuda.to_gpu(y, args.gpu)

            chainer.config.train = True
            model.cleargrads()
            train_loss = model(Variable(x), Variable(y))
            train_loss.backward()
            train_loss.unchain_backward()
            optimizer.update()
            sum_loss += float(cuda.to_cpu(train_loss.data))
            chainer.config.train = False
            valid_loss = float(cuda.to_cpu(model(Variable(v_x), Variable(v_y)).data))

            end = time.time()
            print("{}/{} (epoch {}), train_loss = {:.3f}, valid_loss = {:.3f}, time/batch = {:.3f}"
                  .format(i, args.nb_epochs * data_loader.num_batches,
                          e, sum_loss / (b + 1), valid_loss, end - start))
            if (e * data_loader.num_batches + b) % args.save_every == 0 and ((e * data_loader.num_batches + b) > 0):
                checkpoint_path = os.path.join(args.model_dir,
                                               'model_{}.npz'.format(i))
                serializers.save_npz(checkpoint_path, model)
                print("model saved to {}".format(checkpoint_path))


if __name__ == '__main__':
    main()
