# python experiments/communicative_efficiency.py --num_samples 10 --num_images 10 --num_reductions 10
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
                'speaker_loss', 'listener_loss', 'context_type',
                'use_feedback' 'i_iter', 'round_num', 'target', 'context',
                'caption', 'scores', 'num_words', 'learning_rate',
                'num_steps', 'targetScore', 'correct'
            ]])

    def writerow(self, ctx, round_num, target, caption, num_words, 
                 scores, targetScore, correct) :
        row = [ctx['speaker_loss'], ctx['listener_loss'], ctx['context_type'], 
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
   for context_type in ['challenge'] : # 'far', 'close',
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
           for speaker_loss in ['fixed', 'SCE', 'LCE', 'LCE+LKL', 'SCE+SKL+LCE'] :
               for listener_loss in ['fixed', 'tied_to_speaker', 'SCE', 'LCE', 
                                     'LCE+LKL', 'SCE+SKL+LCE', 'SCE+LCE+LKL'] :
                   grid.append(dict(
                       context_info,
                       speaker_loss = speaker_loss,
                       listener_loss = listener_loss,
                       dirs = img_dirs,
                       cats = img_tags,
                       sample_num = sample_num
                   ))
   return grid

def get_caption(speaker, ctx, cap_key, utt_store) :
    round_num = cap_key.split('-')[-1]
    if ctx['use_feedback'] :
        cap = np.array(speaker.generate_utterance(as_string = False))
    elif cap_key in utt_store :
        cap = utt_store[cap_key]
    else :
        cap = np.array(speaker.generate_utterance(as_string = False))
        utt_store[cap_key] = cap
        if ctx['speaker_loss'] != 'fixed' :
            speaker.update_model(round_num, str_cap)
    return cap, utils.ids_to_words(cap, speaker.vocab)
            
def main(args):
    path = '../data/model_output/listener_accuracy_vs_num_words.csv'
    writer = EfficiencyWriter(args, path)

    # init separate speaker/listener models
    speaker = AdaptiveAgent(args)
    listener = AdaptiveAgent(args)
    grid = construct_expt_grid(args)
    utt_store = {}

    for ctx in grid:
        print("\ntype: {}, speaker loss: {}, listener loss: {}"
              .format(ctx['context_type'], ctx['speaker_loss'], ctx['listener_loss']))

        speaker.loss = ctx['speaker_loss']
        speaker.reset_to_initialization(ctx['dirs'])
        speaker.context_type = ctx['context_type']
        listener.loss = ctx['listener_loss']
        listener.reset_to_initialization(ctx['dirs'])
        listener.context_type = ctx['context_type']

        # update round-robin style by looping through targets in random order
        for round_num in range(1, args.num_reductions) :
            print("")
            targets = random.sample(ctx['dirs'], len(ctx['dirs']))
            for target in targets :
                print('round {}, target {}'.format(round_num, target))

                # set up for new round
                cap_key = "{}-{}-{}-{}".format(
                    ctx['speaker_loss'], ctx['sample_num'], target, round_num
                )
                target_idx = ctx['dirs'].index(target)
                speaker.set_image(target, target_idx)
                listener.set_image(target, target_idx)

                # generate caption and update (reusing if appropriate...)
                cap, str_cap = get_caption(speaker, ctx, cap_key, utt_store)

                # evaluate caption & update listener models as relevent
                scores = listener.L0_score(np.expand_dims(cap, axis=0), ctx['dirs'])
                if not ctx['listener_loss'] in ['fixed', 'tied_to_speaker']:
                    listener.update_model(round_num, str_cap)
                elif ctx['listener_loss'] == 'tied_to_speaker':
                    listener.decoder.load_state_dict(speaker.decoder.state_dict())

                # extract info from listener
                scores = scores.data.cpu().numpy()[0]
                target_score = scores[target_idx]
                listener_idx = list(scores).index(max(scores))
                correct = listener_idx == target_idx

                if ctx['speaker_loss'] != 'fixed' and ctx['use_feedback'] :
                    speaker.set_image(ctx['dirs'][listener_idx], listener_idx)
                    speaker.update_model(round_num, str_cap)

                # Write out
                writer.writerow(ctx, round_num, target, str_cap, len(cap), 
                                scores, target_score, correct)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
    parser.add_argument('--exp_dir', type=str, default = './experiments')
    parser.add_argument('--encoder_path', type=str, default='/share/data/conventions/models/encoder-5-3000.pkl', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='/share/data/conventions/models/decoder-5-3000.pkl', help='path for trained decoder')
    parser.add_argument('--image', type=str, help='input image for generating caption')
    parser.add_argument('--model_path', type=str, default='models/' , help='path for saving trained models')
    parser.add_argument('--vocab_path', type=str, default='/share/data/conventions/data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='/share/data/conventions/data/resized_val2014', help='directory for resized images')
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
    parser.add_argument('--listener_KL_weight', type=float, default=.5)
    parser.add_argument('--listener_CE_weight', type=float, default=1)
    parser.add_argument('--num_rehearsals', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--num_steps', type=int, default=8, help='number of steps to take')
    parser.add_argument('--batch_size', type=int, default=2)
    args = parser.parse_args()

    print(args)
    main(args)
