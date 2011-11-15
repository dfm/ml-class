#!/usr/bin/env python
# encoding: utf-8
"""


"""

from __future__ import division

__all__ = ['']

import numpy as np
import pylab as pl

from mixtures import MixtureModel
from dataset import ImageDataset
from dataset._tile_helper import from_tiles

def kmeans():
    img   = ImageDataset('dataset/buildings.png')
    model = MixtureModel(256, np.array(img.tiles, dtype=np.float64))
    model.run_kmeans(tol=1e-4)
    data = np.zeros(img.shape)

    from_tiles(model.means, model._kmeans_rs, data, (8,8))

    pl.imshow(data, cmap='gray')
    pl.show()

if __name__ == '__main__':
    kmeans()

