import numpy as np
import sys
sys.path.append("../models/")

import argparse
import os
from model import AdaptiveAgent
from utils import Writer
import time
import csv
import context_loader
from PIL import Image
import torchvision.datasets as dset

from random import shuffle
import torchvision.transforms as transforms
from CatastrophicForgettingTester import CatastrophicForgettingExpt
import data_loader as dl
import copy
import utils

class FixCFWriter(Writer) :
    def __init__(self, a, save_file) :
        super().__init__(a, save_file)
        self.learning_rate = a.learning_rate
        self.metric = a.metric
        self.batch_size = a.batch_size
        self.num_steps = a.num_steps
        self.debug = a.debug
        if not self.debug :
            self.init_output_csv([[
                'iter_num', 'control_num',
                'rehearsal_batch_size', 'KL_weight',
                'target_img_dir', 'control_img_dir',
                'control_caption', 'control_caption_length',
                'target_caption','target_caption_length',
                'target_reduction',
                'ctrl_score', 'target_score',
                'metric', 'learning_rate', 'batch_size', 'num_steps'
            ]])

    def writerow(self, ctx, CF, ctrl_score, target_score, target_reduction) :
        row = [
            ctx['iter_num' ], ctx['control_num'],
            ctx['rehearsal_batch_size'], ctx['KL_weight'],
            ctx['target_img_dir'], ctx['control_img_dir'],
            ctx['control_caption'], len(ctx['control_caption']),
            ctx['target_caption'], len(ctx['target_caption']),
            target_reduction,
            ctrl_score, target_score,
            self.metric, self.learning_rate, self.batch_size, self.num_steps,
        ]
        if not self.debug :
            super().writerow(row)

def construct_grid(args) :
    cat_names = utils.get_cat_names()
    grid = []
    for i_iter in range(args.num_targets):
        # pick target & control category to hold out for 'random' vs. 'similar' tests
        # TODO: generalize this to hold out k control_categories instead of 1
        target_cat, control_cat = np.random.choice(cat_names, 2, replace=False)
        heldout_cats = [target_cat, control_cat]

        # sample target from target cat
        targets = utils.choose_similar_images(1, target_cat)[0]

        # choose control images from heldout control category(s)
        controls = utils.choose_similar_images(args.num_control, control_cat)
        assert len(controls) == args.num_control
        for KL_weight in [0, .05, .1, .3, .6, 1.5] :
            for i, cdc2 in enumerate(controls):
                grid.append(dict(
                    iter_num = i_iter,
                    KL_weight = KL_weight,
                    rehearsal_batch_size = rehearsal_batch_size,
                    control_num = i,
                    control_img_dir = cdc2[0],
                    heldout_cats = heldout_cats,
                    control_caption = cdc2[1],
                    target_img_dir = targets[0],
                    target_caption = targets[1],
                ))
    return grid

def main(args):
    writer = FixCFWriter(args, '../data/model_output/KL_rehearsal_nocontext_{}.csv'
                         .format(args.ds_type))

    # Initialize Expt class
    print('initializing CF')
    CF = CatastrophicForgettingExpt(args)

    # Build grid of parameters/images we're going to run experiment over
    print('initializing param grid...')
    grid = construct_grid(args)

    for ctx in grid:
        print(ctx)

        # Reset experiment w/ curr ctx parameters
        CF.reset(ctx['target_img_dir'], ctx)

        # Run reduction, save both initial and final form of model...
        target_reduction = CF.manipulation()

        # Evaluate control caption under initial & final
        ctrl_score = CF.evaluate(ctx['control_caption'], ctx['control_img_dir'])

        # Evaluate target caption under initial & final
        target_score = CF.evaluate(ctx['target_caption'], ctx['target_img_dir'])

        # Save out data
        writer.writerow(ctx, CF, ctrl_score, target_score, target_reduction)

if __name__=='__main__':
    from utils import Vocabulary
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
    parser.add_argument('--exp_dir', type=str, default = './experiments')
    parser.add_argument('--encoder_path', type=str, default='/share/data/conventions/models/encoder-5-3000.pkl', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='/share/data/conventions/models/decoder-5-3000.pkl', help='path for trained decoder')
    parser.add_argument('--model_path', type=str, default='models/' , help='path for saving trained models')
    parser.add_argument('--vocab_path', type=str, default='/share/data/conventions/data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='/share/data/conventions/data/resized_val2014', help='directory for resized images')
    parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
    parser.add_argument('--val_step', type=int , default=10, help='step size for prining val info')
    parser.add_argument('--save_step', type=int , default=1000, help='step size for saving trained models')
    parser.add_argument('--num_workers', type=int, default=0)

    # Expt-specific parameters
    parser.add_argument('--context_size', type=int, default=4)
    parser.add_argument('--num_control', type=int, default=1)
    parser.add_argument('--num_targets', type=int, default=50)
    parser.add_argument('--metric', type=str, default='likelihood')
    parser.add_argument('--debug', type=bool, default=True)

    # Important hyperparams
    parser.add_argument('--num_reductions', type=int, default=10, help='# times to reduce')
    parser.add_argument('--num_rehearsals', type=int, default=10, help='batch size used for rehearsals')
    parser.add_argument('--ds_type', type=str, default='powerset', help='type of dataset')
    parser.add_argument('--loss', type=str, default='KL')
    parser.add_argument('--KL_weight', type=str, default=0)
    parser.add_argument('--num_steps', type=int, default=8, help='number of steps to take')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()


    print(args)
    main(args)

    #lr = 0.0005, batch_size = 4, num_steps = 4


