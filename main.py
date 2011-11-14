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

def kmeans():
    img   = ImageDataset('dataset/buildings.png')
    model = MixtureModel(24, img._tiles)
    model.run_kmeans()
    data = np.zeros(img._shape)
    shape = img._shape
    print np.dot(model._kmeans_rs, model._means.T).shape
    data = np.dot(model._kmeans_rs, model._means.T)
    pl.imshow(data)
    pl.show()

if __name__ == '__main__':
    kmeans()

