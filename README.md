# conventions

* implementation of continual learning for repeated reference is in `/models/`
* interactive, web-based reference game w/ model is in `/behavioral_experiments/`
* experiments on model behavior are in `/listener_experiments/` and `/speaker_experiments/`
* csv output from both kinds of experiments goes in `/data/`
* scripts for analyzing/visualizing data from experiments are in `/analysis/`
* paper is in `/writing/`

# To reproduce experiments

1. install PyTorch and the COCO API:

```
conda install pytorch torchvision -c pytorch
conda install Cython nltk
git clone https://github.com/pdollar/coco.git
cd coco/PythonAPI/
make
python setup.py build
python setup.py install
cd ../../
``` 

2. clone repo and download coco annotations and images for experiments

```
git clone https://github.com/hawkrobe/continual-adaptation.git
sh download-coco.sh
```

3. Extract [pretrained model weights](https://www.dropbox.com/s/ne0ixz5d58ccbbz/pretrained_model.zip?dl=0) and [vocabulary](https://www.dropbox.com/s/26adb7y9m98uisa/vocap.zip?dl=0) files to `./data/preprocess/`

For more information on pretrained model, see tutorial [here](https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning)

# Dependencies

Model code depends on PyTorch >=1.2.0
