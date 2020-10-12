import numpy as np
import sys
sys.path.append("../models/")
import argparse
import os
from model import AdaptiveAgent
import utils
from utils import Writer
from utils import Vocabulary, choose_diff_images

import time
import csv
import context_loader
from PIL import Image
from random import shuffle
import torchvision.transforms as transforms
import data_loader as dl
import torch
import copy

class CFWriter(Writer) :
    def __init__(self, a, save_file) :
        super().__init__(a, save_file)
        self.learning_rate = a.learning_rate
        self.debug = a.debug
        if not self.debug :
            header = [['control_num', 'target_img_dir', 'control_img_dir',
                       'pretest_caption', 'pretest_caption_length', 'posttest_caption','posttest_caption_length', 'ctrl_score',
                'step_size', 'batch_size', 'learning_rate', 'target_score']]
            self.init_output_csv(header)

    def writerow(self, ctx, CF, ctrl_score, target_score) :
        row = [
            ctx['control_num'], ctx['target_img_dir'], ctx['control_img_dir'],
            CF.pre_cap, len(CF.pre_cap), CF.post_cap, len(CF.post_cap), ctrl_score,
            ctx['step_val'], ctx['batch_size'], ctx['learning_rate'], target_score
        ]
        if not self.debug :
            super().writerow([row])


class CatastrophicForgettingExpt():
    """
    Contains functions and variables related to the catastrophic forgetting experiment
    """
    def __init__(self, args):
        self.args = args

        # Initialize an 'orig_agent' we'll use to score
        self.orig_agent = AdaptiveAgent(self.args)

        # Initialize another 'agent' we'll be fine-tuning
        self.agent = AdaptiveAgent(self.args)

    def reset(self, target_img, ctx=None):
        self.target = target_img
        self.agent.reset_to_initialization(target_img)
        if ctx and 'KL_weight' in ctx :
            self.agent.KL_weight = ctx['KL_weight']
        if ctx and 'rehearsal_batch_size' in ctx :
            self.agent.num_rehearsals = ctx['rehearsal_batch_size']
        else:
            self.agent.num_rehearsals = 0

    def check_weight_eq(self, m1, m2):
        for key in m1.state_dict().keys():
            assert(torch.all(torch.eq(m1.state_dict()[key], m2.state_dict()[key])))

    def manipulation(self, context=None, target_img_dir=None):
        if self.args.debug: print("\n==============mainpulation===============")

        # Create initial round's utterance, then start fine-tuning
        caption = self.agent.generate_utterance()
        initial_caption_length = len(caption.split())
        for round_num in range(1, self.args.num_reductions) :
            caption = self.agent.iterate_round(
                round_num, caption,
                context=context,
                target_img_dir=target_img_dir,
            )
            if self.args.debug: print(caption, len(caption)-2)
        # return reduction...
        return len(caption.split()) - initial_caption_length

    def evaluate(self, caption, img):
        t_caption = utils.caption_to_tensor(caption, self.agent.vocab)
        return self._likelihood(t_caption, img)

    def _listener_likelihood(self, caption, img_dir):
        """
        Is the target image just as likely as after the fine-tuning as it was
        before
        """
        pass

    def _likelihood(self, caption, img_dir):
        """
        Does the original caption become less likely after the fine-tuned model
        """
        pre_score = self.orig_agent.S0_score(img_dir, caption, len(caption))
        post_score = self.agent.S0_score(img_dir, caption, len(caption))

        if self.args.debug: print("pre_score, post_score, log ratio",
                                  pre_score, post_score, post_score - pre_score)
        return post_score - pre_score

    def _length_metric(self, c1, c2):
        """
        Calculates the percent difference in number of words from c1 to c2
        (len(c1) - len(c2))/len(c1)
        if len(c2) > len(c1), then
        """
        diff = float(max(0, len(c1)-len(c2)))
        return diff/len(c1)


    def _levenshtein(self, seq1, seq2):
        size_x = len(seq1) + 1
        size_y = len(seq2) + 1
        matrix = np.zeros ((size_x, size_y))
        for x in range(size_x):
            matrix [x, 0] = x
        for y in range(size_y):
            matrix [0, y] = y

        for x in range(1, size_x):
            for y in range(1, size_y):
                if seq1[x-1] == seq2[y-1]:
                    matrix [x,y] = min(
                        matrix[x-1, y] + 1,
                        matrix[x-1, y-1],
                        matrix[x, y-1] + 1
                    )
                else:
                    matrix [x,y] = min(
                        matrix[x-1,y] + 1,
                        matrix[x-1,y-1] + 1,
                        matrix[x,y-1] + 1
                    )
        return (matrix[size_x - 1, size_y - 1])

    def _edit(self, str1, str2, m, n):
        # Create a table to store results of subproblems
        dp = [[0 for x in range(n+1)] for x in range(m+1)]

        # Fill d[][] in bottom up manner
        for i in range(m+1):
            for j in range(n+1):

                # If first string is empty, only option is to
                # insert all characters of second string
                if i == 0:
                    dp[i][j] = j    # Min. operations = j

                # If second string is empty, only option is to
                # remove all characters of second string
                elif j == 0:
                    dp[i][j] = i    # Min. operations = i

                # If last characters are same, ignore last char
                # and recur for remaining string
                elif str1[i-1] == str2[j-1]:
                    dp[i][j] = dp[i-1][j-1]

                # If last character are different, consider all
                # possibilities and find minimum
                else:
                    dp[i][j] = 1 + min(dp[i][j-1],        # Insert
                                       dp[i-1][j],        # Remove
                                       dp[i-1][j-1])    # Replace

        return float(dp[m][n])/max(m,n)


def construct_grid(args) :
    print('constructing grid...')
    # make control and target image same for all hyper param sweeps
    controls, target, _ = dl.choose_control_target_imgs(args)
    grid = []
    # Loop through, sample contexts w/ desired nesting properties
    for step_val in [1,2,4,8,16,32]:
        for bs in [2,4,8,16]:
            for lr in [0.001, 0.0005, 0.0001, 0.00001]:
                for i, control in enumerate(controls):
                    grid.append(dict(step_val = step_val,
                        batch_size = bs,
                        learning_rate = lr,
                        control_num = i,
                        control_img_dir = control,
                        target_img_dir = target
                        ))
    print('made grid')
    return grid


def hyperparam_sweep(args):
    save_file = '../data/model_output/test_cat_forgetting_hyperparam_{}.csv'.format(args.ds_type)
    writer = CFWriter(args, save_file)
    for ctx in construct_grid(args):
        print(ctx)
        args.learning_rate = ctx['learning_rate']
        args.batch_size = ctx['batch_size']
        args.num_steps = ctx['step_val']

        agent = AdapativeAgent(args)
        #CF = CatastrophicForgettingExpt(
        #    args, metric=args.metric,
        #    target=ctx['target_img_dir'], speaker=agent
        #)
        CF = CatastrophicForgettingExpt(args)
        CF.reset(ctx['target_img_dir'])
        CF.manipulation()

        ctrl_score = CF.posttest(ctx['control_img_dir'])
        writer.writerow(ctx, CF, ctrl_score, target_score)

def main(args):
    if args.expt=="hyperparam":
        hyperparam_sweep(args)
    else:
        # make control and target image same for all hyper param sweeps
        #controls, target, _ = dl.choose_control_target_imgs(args)

        imgs = choose_diff_images(100)
        controls, (target_dir, target_cap, target_id) = imgs[:-1], imgs[-1]

        speaker = AdaptiveAgent(args)
        #CF = CatastrophicForgettingExpt(args, metric=args.metric,
        #                                target=target, speaker=speaker)
        CF = CatastrophicForgettingExpt(args)
        CF.reset(target_dir)
        CF.manipulation()

        for (control_dir, control_cap, control_id) in controls:
            CF.evaluate(control_cap, control_dir)
            CF.evaluate(target_cap, target_dir)
            #CF.check_weight_eq(CF.agent.decoder, CF.l2.decoder)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
    parser.add_argument('--exp_dir', type=str, default = './experiments')
    parser.add_argument('--encoder_path', type=str, default='/share/data/conventions/models/encoder-5-3000.pkl', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='/share/data/conventions/models/decoder-5-3000.pkl', help='path for trained decoder')
    parser.add_argument('--model_path', type=str, default='models/' , help='path for saving trained models')
    parser.add_argument('--vocab_path', type=str, default='/share/data/conventions/data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='/share/data/conventions/data/resized2014', help='directory for resized images')
    parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
    parser.add_argument('--val_step', type=int , default=10, help='step size for prining val info')
    parser.add_argument('--save_step', type=int , default=1000, help='step size for saving trained models')
    parser.add_argument('--num_workers', type=int, default=0)

    # Expt-specific parameters
    parser.add_argument('--context_size', type=int, default=4)
    parser.add_argument('--num_control', type=int, default=50)
    parser.add_argument('--expt', type=str)
    parser.add_argument('--metric', type=str, default='likelihood')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--debug', type=bool, default=True)

    # Important hyperparams
    parser.add_argument('--num_reductions', type=int, default=8, help='# times to reduce')
    parser.add_argument('--ds_type', type=str, default='powerset', help='type of dataset')
    parser.add_argument('--loss', type=str, default='KL')
    parser.add_argument('--KL_weight', type=str, default=0.1)
    parser.add_argument('--num_steps', type=int, default=8, help='number of steps to take')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.001)

    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(args)
    main(args)
    # batch size =2, num_steps = 8, learning rate = 0.0001
