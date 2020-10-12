# conventions

* implementation of continual learning for repeated reference is in `/models/`
* interactive, web-based reference game w/ model is in `/behavioral_experiments/`
* experiments on model behavior are in `/computational_experiments/`
* csv output from both kinds of experiments goes in `/data/`
* scripts for analyzing/visualizing data from experiments are in `/analysis/`
* paper is in `/writing/`

# Setup

1. Follow instructions [here](https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning) to download COCO dataset, and install COCO API.

2. Extract [pretrained model weights](https://www.dropbox.com/s/ne0ixz5d58ccbbz/pretrained_model.zip?dl=0) and [vocabulary](https://www.dropbox.com/s/26adb7y9m98uisa/vocap.zip?dl=0) files to `/data/preprocess/`

Place 

# Dependencies

Model code depends on PyTorch >=1.2.0