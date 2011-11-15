#!/usr/bin/env python
# encoding: utf-8

from distutils.core import setup
from distutils.extension import Extension
import numpy.distutils.misc_util

tile_ext = Extension('dataset._tile_helper',
                ['dataset/_tile_helper.c'])

algorithms_ext = Extension('mixtures._algorithms',
                ['mixtures/_algorithms.c'])

setup(packages=['dataset', 'mixtures'],
        ext_modules = [tile_ext, algorithms_ext],
        include_dirs = numpy.distutils.misc_util.get_numpy_include_dirs())

