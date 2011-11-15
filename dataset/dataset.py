#!/usr/bin/env python
# encoding: utf-8
"""
Dataset interface module

Based on dataset.lsh provided by Prof. LeCun

"""

from __future__ import division

__all__ = ['ImageDataset']

import numpy as np

from PIL import Image

from _tile_helper import to_tiles, from_tiles

class ImageDataset(object):
    def __init__(self, fn, tile_shape=(8,8)):
        img = Image.open(fn)
        self.shape = img.size[::-1] # PIL uses transposed shape
        self.data  = np.array(img.getdata()).reshape(self.shape)
        self.tile_size = np.prod(tile_shape)
        self.ntiles = np.prod(self.shape)/self.tile_size
        self.tiles = np.zeros((self.ntiles, self.tile_size), dtype=int)
        to_tiles(self.data, self.tiles, tile_shape)

if __name__ == '__main__':
    img = ImageDataset('dataset/buildings.png')
    import pylab as pl
    pl.imshow(img.tiles)
    pl.show()

