#!/usr/bin/env python

import time

import numpy as np
np.random.seed(110)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl
from mpl_toolkits.axes_grid1 import ImageGrid

from mixtures import MixtureModel
from dataset import ImageDataset

def kmeans():
    fig   = pl.figure(figsize=(10,11))
    grid = ImageGrid(fig, 111, nrows_ncols = (3, 2), axes_pad = 0.05, label_mode = 'L')

    for fn in ['airplane', 'bird', 'boat', 'buildings']:
        print "Dataset: ", fn
        print "=======  ", "="*len(fn)
        img   = ImageDataset('dataset/%s.dat'%fn)

        grid[0].imshow(img.data, cmap='gray')
        grid[0].text(750, 50, 'Raw', fontsize=20, color='r',
                    horizontalalignment='right', verticalalignment='top',)

        for i,k in enumerate([2,4,8,64,256]):
            print "K =", k
            model = MixtureModel(k, np.array(img.tiles, dtype=np.float64))
            model.run_kmeans()
            print "S =", model.get_entropy(), "/", model.get_max_entropy()
            reconstruction = img.reconstruct(model.means, model.responsibilities)

            ax = grid[i+1]
            ax.imshow(reconstruction, cmap='gray')
            ax.text(750, 50, 'K = %d'%k, fontsize=20, color='r',
                    horizontalalignment='right', verticalalignment='top',)
            print

            if k == 8:
                print "Running EM..."
                strt = time.time()
                model.run_em(regularization=1e-10)
                print time.time()-strt, "seconds"
                print

        pl.savefig('results/%s.png'%fn)

if __name__ == '__main__':
    kmeans()

