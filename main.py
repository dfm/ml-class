#!/usr/bin/env python
# encoding: utf-8
"""


"""

from __future__ import division

import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl
from mpl_toolkits.axes_grid1 import ImageGrid

from mixtures import MixtureModel
from dataset import ImageDataset
from dataset._tile_helper import from_tiles

def kmeans():
    fig   = pl.figure(figsize=(10,11))
    grid = ImageGrid(fig, 111, nrows_ncols = (3, 2), axes_pad = 0.05, label_mode = 'L')

    for fn in ['airplane', 'bird', 'boat', 'buildings']:
        print "Dataset: ", fn
        img   = ImageDataset('dataset/%s.png'%fn)

        grid[0].imshow(img.data, cmap='gray')
        grid[0].text(750, 50, 'Raw', fontsize=20, color='r',
                    horizontalalignment='right', verticalalignment='top',)

        for i,k in enumerate([2,4,8,64,256]):
            model = MixtureModel(k, np.array(img.tiles, dtype=np.float64))
            model.run_kmeans(tol=1e-4)
            print model.get_entropy() * img.ntiles
            data = np.zeros(img.shape)
            from_tiles(model.means, model._kmeans_rs, data, (8,8))

            ax = grid[i+1]
            ax.imshow(data, cmap='gray')
            ax.text(750, 50, 'K = %d'%k, fontsize=20, color='r',
                    horizontalalignment='right', verticalalignment='top',)

            if k == 8:
                model.run_em()

        pl.savefig('results/%s.png'%fn)

if __name__ == '__main__':
    kmeans()

