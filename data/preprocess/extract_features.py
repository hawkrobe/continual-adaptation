import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable

from sklearn.decomposition import PCA

from glob import glob
import os

import numpy as np
import pandas as pd
import json
import re

from PIL import Image
import base64

from embeddings import *

'''
To extract features, run: extract_features.sh

or run, e.g.:

python extract_features.py --model coco_encoder --data='/data/rxdh/conventions_data/resized_val2014/' --data_type='images' --spatial_avg=False --ext 'jpg' --out_dir='/data/rxdh/conventions_data/features'

python extract_features.py --model vgg --data='/data/rxdh/conventions_data/resized_val2014/' --layer_ind=5 --data_type='images' --spatial_avg=False --ext 'jpg' --out_dir='/data/rxdh/conventions_data/features'
'''

# retrieve sketch paths
def list_files(path, ext='png'):
    result = [y for x in os.walk(path) for y in glob(os.path.join(x[0], '*.%s' % ext))]
    return result

def check_invalid_sketch(filenames,invalids_path='drawings_to_exclude.txt'):    
    if not os.path.exists(invalids_path):
        print('No file containing invalid paths at {}'.format(invalids_path))
        invalids = []        
    else:
        x = pd.read_csv(invalids_path, header=None)
        x.columns = ['filenames']
        invalids = list(x.filenames.values)
    valids = []   
    basenames = [f.split('/')[-1] for f in filenames]
    for i,f in enumerate(basenames):
        if f not in invalids:
            valids.append(filenames[i])
    return valids

def save_features(features, meta, args):
    features_fname = '' 
    if(args.model == 'vgg') :
        layers = ['P1','P2','P3','P4','P5','FC6','FC7']
        layer_name = layers[int(args.layer_ind)]
        features_fname = 'FEATURES_vgg_{}'.format(layer_name)
    else :
        features_fname = 'FEATURES_coco_encoder'
    np.save(os.path.join(args.out_dir,'{}.npy'.format(features_fname)), 
            features)
    np.savetxt(os.path.join(args.out_dir,'{}.txt'.format(features_fname)), 
               features, delimiter=',')
    meta.to_csv(os.path.join(args.out_dir,'METADATA.csv'), index=True, 
                index_label='feature_ind')
    print('Saved features out to {}!'.format(args.out_dir))

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

if __name__ == "__main__":
    import argparse
    proj_dir = os.path.abspath('../..')
    sketch_dir = os.path.abspath(os.path.join(proj_dir,'sketches'))
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='full path to images', \
                        default=os.path.join(sketch_dir,'combined'))
    parser.add_argument('--layer_ind', help='fc6 = 5, fc7 = 6', default=5)
    parser.add_argument('--num_pcs', help='number of principal components', default=512)    
    parser.add_argument('--data_type', help='"images" or "sketch"', default='images')
    parser.add_argument('--out_dir', help='path to save features to', default='/data/jefan/graphical_conventions/features')    
    parser.add_argument('--spatial_avg', type=bool, help='collapse over spatial dimensions, preserving channel activation only if true', default=True) 
    parser.add_argument('--channel_norm', type=str2bool, help='apply channel-wise normalization?', default='True')    
    parser.add_argument('--test', type=str2bool, help='testing only, do not save features', default='False')  
    parser.add_argument('--ext', type=str, help='image extension type (e.g., "png")', default="png")    
    parser.add_argument('--model', type=str, help='which model to use (raw vgg or our coco encoder)', default="vgg")    

    args = parser.parse_args()
    print('Spatial averaging is {}'.format(args.spatial_avg))
    print('Channel norm is {}'.format(args.channel_norm))
    print('Testing mode is {}'.format(args.test))
    print('Extracting from {}'.format(args.model))
    if(args.model == 'vgg') :
        print('VGG layer index is {}'.format(args.layer_ind))
    print('Num principal components = {}'.format(args.num_pcs))
    
    ## get list of all sketch paths
    image_paths = sorted(list_files(args.data,args.ext))
    print('Length of image_paths before filtering: {}'.format(len(image_paths)))
        
    ## extract features
    extractor = FeatureExtractor(image_paths,
                                 model = args.model,
                                 layer = args.layer_ind,
                                 data_type = args.data_type,
                                 spatial_avg = args.spatial_avg)
    features, paths = extractor.extract_feature_matrix()   
    print(features)
    meta = pd.DataFrame({'img_path' : list(extractor.flatten_list(paths))})
    if args.test==False:        
        save_features(features, meta, args)
