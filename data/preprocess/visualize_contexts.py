import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

def plot_contexts(encoder_name, context_type="hard") :
    # load list of contexts
    d = pd.read_json('./{}_contexts_{}.json'.format(encoder_name, context_type))
    # for each context, plot images in a little grid
    for index, row in d.iterrows():
        ctx = row['neighbor_names']
        cluster_id = row['cluster_ids']

        plt.figure(figsize = (20,4))
        gs1 = gridspec.GridSpec(1, 5)
        gs1.update(wspace=0.025, hspace=0.05)

        url_prefix = '/data/rxdh/conventions_data/resized_val2014/'
        for (i,c) in enumerate(ctx):
            p = plt.subplot(1,5,i+1)
            URL = url_prefix + c
            with Image.open(URL) as img :
                p.imshow(img)
            p.get_xaxis().set_ticklabels([])
            p.get_yaxis().set_ticklabels([])
            p.get_xaxis().set_ticks([])
            p.get_yaxis().set_ticks([])
            p.set_aspect('equal')
            subord = c.split('_')[2]
        plt.title(cluster_id)
        plt.tight_layout()
        plt.savefig('./gallery/{}_{}/context_{}.jpg'.format(encoder_name, context_type, cluster_id))
        plt.close()
