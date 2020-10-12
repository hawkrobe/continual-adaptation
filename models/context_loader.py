import numpy as np
from PIL import Image

import torch
import torch.utils.data as data

from utils import coco, data_dir, data_type
import utils

import json

with open("../data/preprocess/coco_contexts_hard.json", "r") as read_file:
    coco_contexts_hard = json.load(read_file)
    possible_ctx_ids_hard = [ctx['cluster_ids'] for ctx in coco_contexts_hard]
with open("../data/preprocess/coco_contexts_easy.json", "r") as read_file:
    coco_contexts_easy = json.load(read_file)
    possible_ctx_ids_easy = [ctx['cluster_ids'] for ctx in coco_contexts_easy]

class Contexts(data.Dataset):
    def __init__(self, ctx_type, ctx_size, num_samples, tag=None):
        """
        Args:
            ctx_type: (far, close, challenge)
            ctx_size: number of images in context
            num_samples: number of contexts we will return
            tag : optionally specify a particular cluster (for 'challenge' type)
                  or coco category (for 'same' type)
        """
        self.ctx_type = ctx_type
        self.ctx_size = ctx_size
        self.num_samples = num_samples
        self.tag = tag

        assert(isinstance(tag, str) if ctx_type == 'same' else True)
        assert(tag >= 0 and tag < 100 if tag and ctx_type == 'challenge' else True)

    def sample_img_from_tag(self, img_tag) :
        """
        Samples img (and meta-data) from provided coco category tag
        """
        cat_id = coco.getCatIds(catNms=img_tag)
        img_id = np.random.choice(coco.getImgIds(catIds=cat_id), 1)
        img_path = utils.get_img_path(coco.loadImgs(int(img_id))[0]['file_name'])
        return utils.load_image(img_path), img_path, img_tag

    def get_ctx_from_tag(self, ctx_tag) :
        """
        Retrieves requested context from coco_contexts.json or coco_contexts_easy.json
        """
        if self.ctx_type == "challenge":
            coco_contexts = coco_contexts_hard
        if self.ctx_type == "easy":
            coco_contexts = coco_contexts_easy
        ctx = next(filter(lambda x: x['cluster_ids'] == ctx_tag, coco_contexts))
        filenames = ctx['neighbor_names'][:self.ctx_size]
        paths = [utils.get_img_path(name) for name in filenames]
        tags = ['custom' + str(ctx_tag) for i in range(self.ctx_size)]
        imgs = [utils.load_image(path) for path in paths]
        return imgs, paths, tags

    def sample_context(self):
        """
        Samples from possible contexts according to class-level criteria
        """
        if self.ctx_type == 'easy' :
            tag = np.random.choice(possible_ctx_ids_easy) if self.tag is None else self.tag
            imgs,paths,tags = self.get_ctx_from_tag(tag)
            if self.ctx_size > len(paths):
                tags_pool = np.random.choice(possible_ctx_ids_easy, int(np.ceil((self.ctx_size-5)/5)))
                for tag in tags_pool:
                    imgs2, paths2, tags2 = self.get_ctx_from_tag(tag)
                    imgs.extend(imgs2)
                    paths.extend(paths2)
                    tags.extend(tags2)
                imgs = torch.cat(imgs[:self.ctx_size])
                paths = paths[:self.ctx_size]
                tags = tags[:self.ctx_size]
            return imgs, paths, tags
        elif self.ctx_type == 'challenge' :
            tag = np.random.choice(possible_ctx_ids_hard) if self.tag is None else self.tag
            imgs,paths,tags = self.get_ctx_from_tag(tag)
            if self.ctx_size > len(paths):
                tags_pool = np.random.choice(possible_ctx_ids_hard, int(np.ceil((self.ctx_size-5)/5)))
                for tag in tags_pool:
                    imgs2, paths2, tags2 = self.get_ctx_from_tag(tag)
                    imgs.extend(imgs2)
                    paths.extend(paths2)
                    tags.extend(tags2)
                imgs = torch.cat(imgs[:self.ctx_size])
                paths = paths[:self.ctx_size]
                tags = tags[:self.ctx_size]
            return imgs, paths, tags
        elif self.ctx_type == 'close' :
            tag = self.tag if self.tag else np.random.choice(utils.get_cat_names(), 1)
            return [self.sample_img_from_tag(img_tag) for img_tag
                    in [tag for i in range(self.ctx_size)]]
        elif self.ctx_type == 'far' :
            all_cat_names = utils.get_cat_names()
            return [self.sample_img_from_tag(img_tag) for img_tag
                    in np.random.choice(all_cat_names, self.ctx_size, replace=True)]
        else :
            raise Exception('unknown ctx_type: {}'.format(self.ctx_type))

    def __getitem__(self, i):
        """
        Sample a context according to criteria in class-level
        """
        #context = self.sample_context()
        #imgs, img_dirs, tags = zip(*context)
        imgs, img_dirs, tags = self.sample_context()
        return imgs, list(img_dirs), list(tags)

    def __len__(self):
        return self.num_samples

def get_context_loader(ctx_type, ctx_size, num_samples, tag=None):
    return iter(Contexts(ctx_type, ctx_size, num_samples, tag=tag))
