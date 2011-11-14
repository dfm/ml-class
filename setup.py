#!/usr/bin/env python
# encoding: utf-8

from distutils.core import setup
from distutils.extension import Extension
import numpy.distutils.misc_util

ext = Extension('dataset._tile_helper',
                ['dataset/_tile_helper.c'])

setup(packages=['dataset'],
        ext_modules = [ext],
        include_dirs = numpy.distutils.misc_util.get_numpy_include_dirs())

