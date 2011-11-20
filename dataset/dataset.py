#!/usr/bin/env python
"""
Dataset interface module

Based on dataset.lsh provided by Prof. LeCun

The heavy lifting is all done using the _tile_helper C-extension (see
dataset/_tile_helper.c).

"""

__all__ = ['ImageDataset']

import numpy as np

from PIL import Image

from _tile_helper import to_tiles, from_tiles

class ImageDataset(object):
    """
    Construct a tiled dataset given a .png file

    Parameters
    ----------
    fn : str
        Path to the .png file.

    tile_shape : tuple, optional
        The shape of the tiles in pixels (default: (8,8))

    """
    def __init__(self, fn, tile_shape=(8,8)):
        img = Image.open(fn)
        self.shape = img.size[::-1] # PIL uses transposed shape
        self.data  = np.array(img.getdata()).reshape(self.shape)
        self.tile_shape = tile_shape
        self.tile_size = np.prod(tile_shape)
        self.ntiles = np.prod(self.shape)/self.tile_size
        self.tiles = np.zeros((self.ntiles, self.tile_size), dtype=int)
        to_tiles(self.data, self.tiles, tile_shape)

    def reconstruct(self, means, rs):
        """
        Reconstruct the compressed image give a list of prototypes and the tile memberships

        Parameters
        ----------
        means : numpy.ndarray (Nprototype, Ndim)
            The matrix of prototypes

        rs : numpy.ndarray (Ntiles,)
            The list of integer memberships for each tile.  Each element of rs
            is is [0,Nprototype).

        Returns
        -------
        reconstruction : numpy.ndarray (same as original image)
            The reconstructed image

        """
        result = np.zeros(self.shape)
        from_tiles(means, rs, result, self.tile_shape)
        return result

if __name__ == '__main__':
    img = ImageDataset('dataset/buildings.png')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as pl

    pl.imshow(img.tiles)
    pl.savefig('tile_example.png')

