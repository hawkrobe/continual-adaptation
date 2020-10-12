# python experiments/get_fixed_points.py --num_samples 10 --num_images 10 --num_reductions 10
import sys
sys.path.append("../models/")

import argparse
import os
from model import AdaptiveAgent
from utils import Writer, coco, data_dir, data_type
from utils import Vocabulary
from tensorboardX import SummaryWriter
import time
import csv

def main(args, img_id=None):
    # write header
    writer = Writer(args, '../data/model_output/ppt_fixed_points_{}.csv'.format(args.ds_type))
    writer.init_output_csv()
    speaker = AdaptiveAgent(args)

    # return path to image with caption
    if img_id:
        img_path = coco.loadImgs(img_id)[0]['file_name']
        image = '{}/resized_{}/{}'.format(data_dir, data_type, img_path)

        for sample_num in range(args.num_samples):
            speaker.reset_to_initialization(image)

            caption = speaker.generate_utterance()
            writer.write(0, caption, sample_num, image)

            for round_num in range(1, args.num_reductions) :
                caption = speaker.iterate_round(round_num, caption)
                writer.write(round_num, caption, sample_num, image)

    else:
        for i, img in enumerate(os.listdir(args.image_dir)):
            if i==args.num_images:
                break
            elif img.endswith(".jpg"):
                print("Image " + str(i) + ": " + img)
                image = args.image_dir+"/"+img

            for sample_num in range(args.num_samples):
                print("iteration: ", sample_num)
                speaker.reset_to_initialization(image)

                caption = speaker.generate_utterance()
                writer.write(0, caption, sample_num, image)

                for round_num in range(1, args.num_reductions) :
                    caption = speaker.iterate_round(round_num, caption)
                    writer.write(round_num, caption, sample_num, image)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
    parser.add_argument('--encoder_path', type=str, default='/share/data/conventions/models/encoder-5-3000.pkl', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='/share/data/conventions/models/decoder-5-3000.pkl', help='path for trained decoder')
    parser.add_argument('--model_path', type=str, default='models/' , help='path for saving trained models')
    parser.add_argument('--vocab_path', type=str, default='/share/data/conventions/data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='/share/data/conventions/data/resized2014', help='directory for resized images')
    parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
    parser.add_argument('--val_step', type=int , default=10, help='step size for prining val info')
    parser.add_argument('--save_step', type=int , default=1000, help='step size for saving trained models')
    parser.add_argument('--num_workers', type=int, default=2)

    # Expt-specific parameters
    parser.add_argument('--num_images', type=int, default=1, help='number of images to test on')
    parser.add_argument('--num_samples', type=int, default=10, help='number of random seeds to re-run each image on')
    parser.add_argument('--context_size', type=int, default=4)
    parser.add_argument('--debug', type=bool, default=True)

    # Important hyperparams
    parser.add_argument('--ds_type', type=str, default='powerset', help='type of dataset')
    parser.add_argument('--num_reductions', type=int, default=8, help='# times to reduce')
    parser.add_argument('--num_rehearsals', type=int, default=10)
    parser.add_argument('--loss', type=str, default='KL')
    parser.add_argument('--KL_weight', type=str, default=0.1)
    parser.add_argument('--num_steps', type=int, default=8, help='number of steps to take')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.001)

    args = parser.parse_args()

    print(args)
    main(args, img_id=144723)
