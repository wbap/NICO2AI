#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Takuma Yagi <tyagi@iis.u-tokyo.ac.jp>
#
# Distributed under terms of the MIT license.

import numpy as np
import random

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable, cuda


class Model(chainer.Chain):

    def __init__(self, nb_layers, nb_units, nb_mixtures, data_scale, dropout):
        NOUT = 1 + nb_mixtures * 6  # end_of_stroke + prob + 2*(mu + sig) + corr
        super(Model, self).__init__(
            fc=L.Linear(nb_units, NOUT)
        )
        for l in range(nb_layers):
            self.add_link("cell_{}".format(l), L.LSTM(None, nb_units))

        self.nb_layers = nb_layers
        self.nb_units = nb_units
        self.nb_mixtures = nb_mixtures
        self.data_scale = data_scale

    def reset_state(self):
        for l in range(self.nb_layers):
            self["cell_{}".format(l)].reset_state()

    def __call__(self, input_data, target_data, predict=False):

        def calc_2d_normal(x1, x2, mu1, mu2, s1, s2, rho):
            norm1 = F.broadcast_to(x1, mu1.shape) - mu1
            norm2 = F.broadcast_to(x2, mu2.shape) - mu2
            s1s2 = s1 * s2
            z = F.square(norm1 / s1) + F.square(norm2 / s2) - 2 * rho * norm1 * norm2 / s1s2
            neg_rho = 1 - F.square(rho)
            return F.exp(-z / (2 * neg_rho)) / (2 * np.pi * s1s2 * F.sqrt(neg_rho))

        def get_loss(z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_eos, x1_data, x2_data, eos_data):
            xp = cuda.get_array_module(x1_data)
            result0 = calc_2d_normal(x1_data, x2_data, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr)

            # implementing eq # 26 of http://arxiv.org/abs/1308.0850
            result1 = F.sum(result0 * z_pi, axis=1, keepdims=True)
            mask = Variable(xp.ones_like(result1, dtype=np.float32) * 1e-20)
            result1 = -F.log(F.maximum(result1, mask))
            result2 = -F.log(z_eos * eos_data + (1-z_eos) * (1-eos_data))

            return F.sum(result1 + result2)

        def get_mixture_coeff(z):
            z_eos = z[:, 0:1]
            z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr = F.split_axis(z[:, 1:], 6, axis=1)

            z_eos = F.sigmoid(z_eos)
            z_pi = F.softmax(z_pi, axis=1)
            z_sigma1 = F.exp(z_sigma1)
            z_sigma2 = F.exp(z_sigma2)
            z_corr = F.tanh(z_corr)
            return [z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_eos]

        self.reset_state()
        inputs = F.separate(input_data, axis=1)
        outputs = []
        for h in inputs:
            for l in range(self.nb_layers):
                h = self["cell_{}".format(l)](h)
            outputs.append(h)

        output = F.reshape(F.concat(outputs, axis=1), (-1, self.nb_units))
        output = self.fc(output)

        [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_eos] = get_mixture_coeff(output)

        if predict:  # Used in function sample
            return list(map(lambda x: cuda.to_cpu(x.data), [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_eos]))
        else:
            flat_target_data = F.reshape(target_data, (-1, 3))
            [x1_data, x2_data, eos_data] = F.split_axis(flat_target_data, 3, axis=1)
            loss = get_loss(o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_eos, x1_data, x2_data, eos_data)
            cost = loss / (input_data.shape[0] * input_data.shape[1])
            return cost

    def sample(self, num=1200, gpu=-1):

        def get_pi_idx(x, pdf):
            N = pdf.size
            accumulate = 0
            for i in range(0, N):
                accumulate += pdf[i]
                if (accumulate >= x):
                    return i
            print('error with sampling ensemble')
            return -1

        def sample_gaussian_2d(mu1, mu2, s1, s2, rho):
            mean = [mu1, mu2]
            cov = [[s1*s1, rho*s1*s2], [rho*s1*s2, s2*s2]]
            x = np.random.multivariate_normal(mean, cov, 1)
            return x[0][0], x[0][1]

        self.reset_state()
        prev_x = np.zeros((1, 1, 3), dtype=np.float32)
        prev_x[0, 0, 2] = 1  # initially, we want to see beginning of new stroke
        if gpu >= 0:
            prev_x = cuda.to_gpu(prev_x, gpu)

        xp = chainer.cuda.get_array_module(prev_x)

        strokes = np.zeros((num, 3), dtype=np.float32)
        mixture_params = []

        for i in range(num):
            [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_eos] = self.__call__(prev_x, None, True)

            # Randomly select one mixture components and sample next point
            idx = get_pi_idx(random.random(), o_pi[0])
            eos = 1 if random.random() < o_eos[0][0] else 0
            next_x1, next_x2 = sample_gaussian_2d(o_mu1[0][idx], o_mu2[0][idx],
                                                  o_sigma1[0][idx], o_sigma2[0][idx], o_corr[0][idx])
            strokes[i, :] = [next_x1, next_x2, eos]
            params = [o_pi[0], o_mu1[0], o_mu2[0], o_sigma1[0], o_sigma2[0], o_corr[0], o_eos[0]]
            mixture_params.append(params)

            prev_x = xp.zeros((1, 1, 3), dtype=np.float32)
            prev_x[0][0] = xp.array([next_x1, next_x2, eos], dtype=np.float32)

        strokes[:, 0:2] *= self.data_scale
        return strokes, mixture_params


class Model_NStepLSTM(chainer.Chain):

    def __init__(self, nb_layers, nb_units, nb_mixtures, data_scale, dropout):
        NOUT = 1 + nb_mixtures * 6  # end_of_stroke + prob + 2*(mu + sig) + corr
        super(Model, self).__init__(
            encoder=L.NStepLSTM(nb_layers, nb_units, nb_units, dropout),
            fc=L.Linear(nb_units, NOUT)
        )

        self.nb_layers = nb_layers
        self.nb_units = nb_units
        self.nb_mixtures = nb_mixtures
        self.data_scale = data_scale

    def reset_state(self):
        self.h, self.c = None, None

    def __call__(self, input_data, target_data, predict=False):

        def calc_2d_normal(x1, x2, mu1, mu2, s1, s2, rho):
            norm1 = F.broadcast_to(x1, mu1.shape) - mu1
            norm2 = F.broadcast_to(x2, mu2.shape) - mu2
            s1s2 = s1 * s2
            z = F.square(norm1 / s1) + F.square(norm2 / s2) - 2 * rho * norm1 * norm2 / s1s2
            neg_rho = 1 - F.square(rho)
            return F.exp(-z / (2 * neg_rho)) / (2 * np.pi * s1s2 * F.sqrt(neg_rho))

        def get_loss(z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_eos, x1_data, x2_data, eos_data):
            xp = cuda.get_array_module(x1_data)
            result0 = calc_2d_normal(x1_data, x2_data, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr)

            # implementing eq # 26 of http://arxiv.org/abs/1308.0850
            result1 = F.sum(result0 * z_pi, axis=1, keepdims=True)
            mask = Variable(xp.ones_like(result1, dtype=np.float32) * 1e-20)
            result1 = -F.log(F.maximum(result1, mask))
            result2 = -F.log(z_eos * eos_data + (1-z_eos) * (1-eos_data))

            return F.sum(result1 + result2)

        def get_mixture_coeff(z):
            z_eos = z[:, 0:1]
            z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr = F.split_axis(z[:, 1:], 6, axis=1)

            z_eos = F.sigmoid(z_eos)
            z_pi = F.softmax(z_pi, axis=1)
            z_sigma1 = F.exp(z_sigma1)
            z_sigma2 = F.exp(z_sigma2)
            z_corr = F.tanh(z_corr)
            return [z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_eos]

        self.reset_state()
        inputs = F.separate(input_data, axis=0)  # Split by batch
        self.h, self.c, outputs = self.encoder(self.h, self.c, inputs)

        output = F.reshape(F.concat(outputs, axis=0), (-1, self.nb_units))
        output = self.fc(output)

        [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_eos] = get_mixture_coeff(output)

        if predict:  # Used in function sample
            return list(map(lambda x: cuda.to_cpu(x.data), [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_eos]))
        else:
            flat_target_data = F.reshape(target_data, (-1, 3))
            [x1_data, x2_data, eos_data] = F.split_axis(flat_target_data, 3, axis=1)
            loss = get_loss(o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_eos, x1_data, x2_data, eos_data)
            cost = loss / (input_data.shape[0] * input_data.shape[1])
            return cost

    def sample(self, num=1200, gpu=-1):
        def get_pi_idx(x, pdf):
            N = pdf.size
            accumulate = 0
            for i in range(0, N):
                accumulate += pdf[i]
                if (accumulate >= x):
                    return i
            print('error with sampling ensemble')
            return -1

        def sample_gaussian_2d(mu1, mu2, s1, s2, rho):
            mean = [mu1, mu2]
            cov = [[s1*s1, rho*s1*s2], [rho*s1*s2, s2*s2]]
            x = np.random.multivariate_normal(mean, cov, 1)
            return x[0][0], x[0][1]

        self.reset_state()
        prev_x = np.zeros((1, 1, 3), dtype=np.float32)
        prev_x[0, 0, 2] = 1  # initially, we want to see beginning of new stroke
        if gpu >= 0:
            prev_x = cuda.to_gpu(prev_x, gpu)

        xp = chainer.cuda.get_array_module(prev_x)

        strokes = np.zeros((num, 3), dtype=np.float32)
        mixture_params = []

        for i in range(num):
            [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_eos] = self.__call__(prev_x, None, True)

            # Randomly select one mixture components and sample next point
            idx = get_pi_idx(random.random(), o_pi[0])
            eos = 1 if random.random() < o_eos[0][0] else 0
            next_x1, next_x2 = sample_gaussian_2d(o_mu1[0][idx], o_mu2[0][idx],
                                                  o_sigma1[0][idx], o_sigma2[0][idx], o_corr[0][idx])
            strokes[i, :] = [next_x1, next_x2, eos]
            params = [o_pi[0], o_mu1[0], o_mu2[0], o_sigma1[0], o_sigma2[0], o_corr[0], o_eos[0]]
            mixture_params.append(params)

            prev_x = xp.zeros((1, 1, 3), dtype=np.float32)
            prev_x[0][0] = xp.array([next_x1, next_x2, eos], dtype=np.float32)

        strokes[:, 0:2] *= self.data_scale
        return strokes, mixture_params
