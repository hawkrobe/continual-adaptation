in order to generate additional contexts:

1. use the `generate_context.ipynb` notebook as an entry point 
2. this notebook calls out to the `make_contexts()` function in `generate_hard_contexts.py` where you can specify whether you want 'vgg' or 'coco' image features.
3. `make_contexts()` relies on already having pre-generated the image features. to generate these features, use `extract_features.py` on GPU

while 'vgg' features can be generated with no dependencies, 'coco' relies on using the visual features from the pre-trained model 
