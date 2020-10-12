import sys
sys.path.append("../models/")

import argparse
import os
from model import AdaptiveAgent
import utils
from utils import Writer, Vocabulary
from utils import coco, choose_similar_images, choose_diff_images
import numpy as np
import context_loader
import random
from random import shuffle
import torch


class EfficiencyWriter(Writer) :
    def __init__(self, a, save_file) :
        super().__init__(a, save_file)
        self.learning_rate = a.learning_rate
        self.num_steps = a.num_steps
        self.debug = a.debug
        if not self.debug :
            self.init_output_csv([[
                'speaker_loss', 'context_type',
                'use_feedback' 'i_iter', 'round_num', 'target', 'context',
                'caption', 'scores', 'num_words', 'learning_rate',
                'num_steps', 'targetScore', 'correct'
            ]])

    def writerow(self, ctx, round_num, target, caption, num_words, 
                 scores, targetScore, correct) :
        row = [ctx['speaker_loss'], ctx['context_type'], 
               ctx['use_feedback'], ctx['sample_num'], round_num, target, ctx['dirs'],
               caption, scores, num_words, self.learning_rate,
               self.num_steps, targetScore, correct]
        if not self.debug :
            super().writerow(row)

def get_context_loader(context_info):
    return context_loader.get_context_loader(
        ctx_type = context_info['context_type'],
        ctx_size=context_info['context_size'],
        num_samples = context_info['num_samples']
    )

def construct_expt_grid(args) :
   # Loop through, sample contexts w/ desired nesting properties
   grid = []
   for context_type in ['easy'] :
       context_info = dict(
           context_type = context_type,
           context_size = args.context_size,
           num_samples = args.num_samples,
           use_feedback = args.use_feedback
       )

       # call context loader
       ctx_loader = get_context_loader(context_info)
       for sample_num in range(args.num_samples) :
           ctx = next(ctx_loader)
           imgs, img_dirs, img_tags = ctx
           for speaker_loss in ['SCE+SKL+SR'] :
               grid.append(dict(
                   context_info,
                   speaker_loss = speaker_loss,
                   dirs = img_dirs,
                   cats = img_tags,
                   sample_num = sample_num
               ))
   return grid

def get_caption(speaker, ctx, cap_key, utt_store) :
    round_num = cap_key.split('-')[-1]
    if ctx['use_feedback'] :
        cap = np.array(speaker.generate_utterance('S0'))
    elif cap_key in utt_store :
        cap = utt_store[cap_key]
    else :
        cap = np.array(speaker.generate_utterance('S0', as_string = False))
        utt_store[cap_key] = cap
    return cap, utils.ids_to_words(cap, speaker.vocab)
            
def main(args):
    path = '../data/model_output/speaker_production.csv'
    writer = EfficiencyWriter(args, path)

    # init separate speaker/listener models
    speaker = AdaptiveAgent(args)
    grid = construct_expt_grid(args)
    utt_store = {}

    for ctx in grid:
        print("\ntype: {}, speaker loss: {}"
              .format(ctx['context_type'], ctx['speaker_loss']))

        speaker.loss = ctx['speaker_loss']
        speaker.reset_to_initialization(ctx['dirs'])
        shuffled_context =random.sample( ctx['dirs'], len(ctx['dirs']))
        speaker.context_type = ctx['context_type']

        # simulate round-robin style by looping through targets in random order
        for round_num in range(1, args.num_reductions) :
            targets = random.sample(ctx['dirs'], len(ctx['dirs']))
            for target in targets :
                print('round {}, target {}'.format(round_num, target))

                # set up for new round
                cap_key = "{}-{}-{}-{}".format(
                    ctx['speaker_loss'], ctx['sample_num'], target, round_num
                )
                speaker.set_image(target)

                # generate caption and update (reusing if appropriate...)
                cap, str_cap = get_caption(speaker, ctx, cap_key, utt_store)

                speaker.set_context(ctx['dirs'])
                scores1 = speaker.L0_score(np.expand_dims(cap, axis=0))
                speaker.set_context(shuffled_context)
                scores2 = speaker.L0_score(np.expand_dims(cap, axis=0))

                # evaluate caption & update listener models as relevent
                if ctx['speaker_loss'] != 'fixed' :
                    str_cap = utils.ids_to_words(cap, speaker.vocab)
                    speaker.update_model(round_num, str_cap)

                speaker.set_context(ctx['dirs'])
                scores1 = speaker.L0_score(np.expand_dims(cap, axis=0))
                speaker.set_context(shuffled_context)
                scores2 = speaker.L0_score(np.expand_dims(cap, axis=0))
                print(scores1)
                print(scores2)
                return

                # extract info from listener
                scores = scores.data.cpu().numpy()[0]
                target_img_idx = speaker.context.index(speaker.target)
                target_score = scores[target_img_idx]
                listener_idx = list(scores).index(max(scores))
                correct = listener_idx == target_img_idx
                print(scores, target_score)
                # Write out
                writer.writerow(ctx, round_num, target, str_cap, len(cap), 
                                scores, target_score, correct)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
    parser.add_argument('--exp_dir', type=str, default = './experiments')
    parser.add_argument('--encoder_path', type=str, default='/data/rxdh/conventions_data/encoder-5-3000.pkl', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='/data/rxdh/conventions_data/decoder-5-3000.pkl', help='path for trained decoder')
    parser.add_argument('--image', type=str, help='input image for generating caption')
    parser.add_argument('--model_path', type=str, default='models/' , help='path for saving trained models')
    parser.add_argument('--vocab_path', type=str, default='/data/rxdh/conventions_data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='/data/rxdh/conventions_data/resized_val2014', help='directory for resized images')

    parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
    parser.add_argument('--val_step', type=int , default=10, help='step size for prining val info')
    parser.add_argument('--save_step', type=int , default=1000, help='step size for saving trained models')
    parser.add_argument('--num_workers', type=int, default=0)

    # Expt-specific parameters
    parser.add_argument('--context_size', type=int, default=4)
    parser.add_argument('--use_feedback', type=bool, default=False)
    parser.add_argument('--context_sim_metric', type=str, default='')
    parser.add_argument('--num_samples', type=int, default=50)
    parser.add_argument('--debug', action='store_true')

    # Important hyper-params
    parser.add_argument('--num_reductions', type=int, default=8, help='# times to reduce')
    parser.add_argument('--ds_type', type=str, default='powerset', help='type of dataset')
    parser.add_argument('--loss', type=str, default='SCE')
    parser.add_argument('--speaker_KL_weight', type=float, default=.5)
    parser.add_argument('--speaker_CE_weight', type=float, default=1)
    parser.add_argument('--speaker_rehearsal_weight', type=float, default=1)
    parser.add_argument('--listener_KL_weight', type=float, default=.5)
    parser.add_argument('--listener_CE_weight', type=float, default=.5)
    parser.add_argument('--listener_rehearsal_weight', type=float, default=1)
    parser.add_argument('--reduction_history_window', type=str, default='complete')
    parser.add_argument('--num_rehearsals', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--num_steps', type=int, default=8, help='number of steps to take')
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    print(args)
    main(args)
