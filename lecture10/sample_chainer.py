#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Takuma Yagi <tyagi@iis.u-tokyo.ac.jp>
#
# Distributed under terms of the MIT license.

import argparse
import os
import pickle

from chainer import serializers

from utils import draw_strokes, draw_strokes_random_color, draw_strokes_eos_weighted, draw_strokes_pdf
from model_chainer import Model


parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, default='sample',
                    help='filename of .svg file to output, without .svg')
parser.add_argument('--sample_length', type=int, default=800,
                    help='number of strokes to sample')
parser.add_argument('--scale_factor', type=int, default=10,
                    help='factor to scale down by for svg output.  smaller means bigger output')
parser.add_argument('--model_dir', type=str, default='save',
                    help='directory to save model to')
parser.add_argument('--model_name', type=str, default='model.npz',
                    help='model file name (npz)')

sample_args = parser.parse_args()

with open(os.path.join(sample_args.model_dir, 'config.pkl'), 'rb') as f:
    saved_args = pickle.load(f)

model = Model(saved_args.nb_layers, saved_args.nb_units,
              saved_args.nb_mixtures, saved_args.data_scale,
              saved_args.keep_prob)

checkpoint_path = os.path.join(saved_args.model_dir, sample_args.model_name)
serializers.load_npz(checkpoint_path, model)
print("loading model: ", checkpoint_path)

if saved_args.gpu >= 0:
    model.to_gpu(saved_args.gpu)

def sample_stroke():
    [strokes, params] = model.sample(sample_args.sample_length, saved_args.gpu)
    draw_strokes(strokes, factor=sample_args.scale_factor,
                 svg_filename=sample_args.filename+'.normal.svg')
    draw_strokes_random_color(strokes, factor=sample_args.scale_factor,
                              svg_filename=sample_args.filename+'.color.svg')
    draw_strokes_random_color(strokes, factor=sample_args.scale_factor,
                              per_stroke_mode=False,
                              svg_filename=sample_args.filename +
                              '.multi_color.svg')
    draw_strokes_eos_weighted(strokes, params,
                              factor=sample_args.scale_factor,
                              svg_filename=sample_args.filename +
                              '.eos_pdf.svg')
    draw_strokes_pdf(strokes, params, factor=sample_args.scale_factor,
                     svg_filename=sample_args.filename+'.pdf.svg')
    return [strokes, params]


[strokes, params] = sample_stroke()
