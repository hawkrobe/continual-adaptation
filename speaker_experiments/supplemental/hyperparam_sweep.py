import sys
import argparse
import os
import time
sys.path.append('../models/')

from model import AdaptiveAgent
from utils import Writer
from utils import coco, choose_similar_images, choose_diff_images
from utils import Vocabulary
import numpy as np

from itertools import product

class HyperWriter(Writer) :
    def __init__(self, a, save_file) :
        super().__init__(a, save_file)
        self.debug = a.debug
        if not self.debug :
            self.init_output_csv([[
                'sample_num', 'learning_rate', 'num_steps', 'batch_size',
                'dataset_type', 'isKL', 'round', 'num_words', 'caption'
            ]])

    def writerow(self, args, sample_num, round_num, caption, is_KL):
        num_words = len(caption.split())-2
        row = [sample_num, args.learning_rate, args.num_steps, args.batch_size,
               args.ds_type, is_KL, round_num, num_words, caption
        ]
        if not self.debug :
            super().writerow(row)
def hyperparam_grid () :
    #param_keys     = ['step_val', 'lr', 'batch_size', 'ds_type']
    #step_vals      = [1,2,4,8,16,32,64]
    #batch_sizes    = [2,4,8,16]
    #learning_rates = [0.0001, 0.001, 0.005]
    #ds_types       = ['powerset', 'NPs']
    #combos = list(product(step_vals, learning_rates, batch_sizes, ds_types))

    param_keys     = ['step_val', 'lr', 'batch_size', 'is_KL']
    step_vals      = [8,16,32,64]
    batch_sizes    = [2,15,32,64]
    learning_rates = [0.001, 0.005, 0.01]
    #ds_types = ['powerset', 'ordered_subset']
    is_KL = [True,False]
    combos = list(product(step_vals, learning_rates, batch_sizes, is_KL))
    return list(map(lambda values: dict( zip(param_keys, values)), combos))

def main(args):
    if not args.debug:
        path = '../data/model_output/hyperparam_sweep.csv'
        writer = HyperWriter(args, path)
    categories = coco.loadCats(coco.getCatIds())
    for hp in hyperparam_grid() :
        print(hp)
        args.num_steps = hp['step_val']
        args.learning_rate = hp['lr']
        args.batch_size = hp['batch_size']
        #args.ds_type = hp['ds_type']
        speaker = AdaptiveAgent(args)
        for i_iter in range(args.num_images):
            print("\nchoosing target")
            target_cat = np.random.choice(categories, 1, replace=False)[0]['name']
            heldout_cats = [target_cat]
            targets = choose_similar_images(1, target_cat)
            target_img_dir, _, _ = targets[0][0], targets[0][1], targets[0][2]
            print("choosing rehearsals")
            if hp['is_KL']:
                rehearsals = choose_diff_images(100, cats_to_avoid=heldout_cats)
                rehearsal_batch_size = 30
            else:
                rehearsals = None
                rehearsal_batch_size = 0
            speaker.reset_to_initialization(target_img_dir)
            round_num = 0
            caption = speaker.generate_utterance()
            if args.debug: print("ORIG CAPTION: ", caption)
            if not args.debug: writer.writerow(args, i_iter, round_num, caption, hp['is_KL'])
            while round_num < args.num_reductions:
                round_num += 1
                if len(caption.split())-2>0: caption = speaker.iterate_round(round_num, caption, all_rehearsals=rehearsals, rehearsal_batch_size=rehearsal_batch_size)
                if args.debug:
                    print(len(caption.split())-2, caption)
                if not args.debug: writer.writerow(args, i_iter, round_num, caption, hp['is_KL'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_path', type=str, default='/share/data/conventions/models/encoder-5-3000.pkl', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='/share/data/conventions/models/decoder-5-3000.pkl', help='path for trained decoder')
    parser.add_argument('--image', type=str, required=False, help='input image for generating caption')
    parser.add_argument('--model_path', type=str, default='../models/' , help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='/share/data/conventions/data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='/share/data/conventions/data/val2014', help='directory for resized images')
    parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
    parser.add_argument('--val_step', type=int , default=10, help='step size for prining val info')
    parser.add_argument('--save_step', type=int , default=1000, help='step size for saving trained models')

    # Model parameters
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--num_images', type=int, default=10)
    parser.add_argument('--context_size', type=int, default=4)
    parser.add_argument('--num_rehearsals', type=int, default=30)
    parser.add_argument('--debug', action='store_true')

    # Important hyperparams
    parser.add_argument('--num_reductions', type=int, default=8, help='# times to reduce')
    parser.add_argument('--ds_type', type=str, default='powerset', help='type of dataset')
    parser.add_argument('--loss', type=str, default='KL')
    parser.add_argument('--KL_weight', type=str, default=0.1)
    parser.add_argument('--num_steps', type=int, default=8, help='number of steps to take')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()

    print(args)
    main(args)
