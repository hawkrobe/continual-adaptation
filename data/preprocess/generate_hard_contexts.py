import sklearn
import numpy as np
import pandas as pd
import sys
sys.path.append("../../models")
import utils

import sklearn.metrics.pairwise as pw
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

# set up apis
#Neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='cosine')
#KMeans = MiniBatchKMeans(n_clusters=100, batch_size=1000,max_no_improvement=20)

def make_hard_contexts(encoder_name):
    # load in features and initialize
    path = '/data/rxdh/conventions_data/features'
    meta = pd.read_csv('{}/METADATA.csv'.format(path))
    filename = 'FEATURES_vgg_FC6' if encoder_name == 'vgg' else 'FEATURES_coco_encoder'
    feats = np.load('{}/{}.npy'.format(path, filename))[:meta.shape[0],]
    print("feats: ", feats)
    contexts = []

    # set up apis
    n_neighbors = 5 if context_type=="hard" else 100  # changing neighbor radius 
    Neighbors = NearestNeighbors(n_neighbors=n_neighbors, algorithm='brute', metric='cosine')
    KMeans = MiniBatchKMeans(n_clusters=100, batch_size=1000,max_no_improvement=20)

    # fit k-nearest neighbors model on features
    neighs = Neighbors.fit(feats)

    # extract clusters from kmeans
    meta['cluster_ids'] = KMeans.fit_predict(feats)

    # for each cluster, sample an image and save its nearest neighbors...
    for name,cluster in meta.groupby('cluster_ids') :
        row = cluster.sample(1)
        row_feats = feats[row.index,:]
        neighbor_indices = neighs.kneighbors(row_feats, return_distance=False)[0]
        neighbor_indices = np.random.choice(neighbor_indices, 5, replace=False)  # sample 5 in case there are more than 5 neighbors
        neighbor_names = [meta.loc[meta.index[i],'img_path'].split('/')[-1] for i in neighbor_indices]
        row = row.assign(neighbor_indices = [neighbor_indices], neighbor_names = [neighbor_names])
        contexts.append(row)
    return contexts

def make_easy_contexts():
    contexts = []
    for i in range(100):
        cat_names = utils.get_cat_names()
        img_ids = [utils.sample_img_from_tag(img_tag)[1].split('/')[-1] for img_tag in np.random.choice(cat_names, 5, replace=False)]
        contexts.append(pd.DataFrame({"cluster_ids":[i], "neighbor_names":[img_ids]}))
    return contexts
    
def make_contexts(encoder_name, context_type="hard") :
    if context_type == "hard":
        contexts = make_hard_contexts(encoder_name)
    elif context_type == "easy":
        contexts = make_easy_contexts()
    pd.concat(contexts).to_json('{}_contexts_{}.json'.format(encoder_name, context_type), orient='records')

def make_contexts_backup(encoder_name, context_type="hard") :
    # load in features and initialize
    path = '/data/rxdh/conventions_data/features'
    meta = pd.read_csv('{}/METADATA.csv'.format(path))
    filename = 'FEATURES_vgg_FC6' if encoder_name == 'vgg' else 'FEATURES_coco_encoder'
    feats = np.load('{}/{}.npy'.format(path, filename))[:meta.shape[0],]
    print("feats: ", feats)
    contexts = []

    # set up apis
    n_neighbors = 5 if context_type=="hard" else 100  # changing neighbor radius 
    Neighbors = NearestNeighbors(n_neighbors=n_neighbors, algorithm='brute', metric='cosine')
    KMeans = MiniBatchKMeans(n_clusters=100, batch_size=1000,max_no_improvement=20)

    # fit k-nearest neighbors model on features
    neighs = Neighbors.fit(feats)

    # extract clusters from kmeans
    meta['cluster_ids'] = KMeans.fit_predict(feats)

    # for each cluster, sample an image and save its nearest neighbors...
    for name,cluster in meta.groupby('cluster_ids') :
        row = cluster.sample(1)
        row_feats = feats[row.index,:]
        neighbor_indices = neighs.kneighbors(row_feats, return_distance=False)[0]
        neighbor_indices = np.random.choice(neighbor_indices, 5, replace=False)  # sample 5 in case there are more than 5 neighbors
        neighbor_names = [meta.loc[meta.index[i],'img_path'].split('/')[-1] for i in neighbor_indices]
        row = row.assign(neighbor_indices = [neighbor_indices], neighbor_names = [neighbor_names])
        contexts.append(row)
    pd.concat(contexts).to_json('{}_contexts_{}.json'.format(encoder_name, context_type), orient='records')
