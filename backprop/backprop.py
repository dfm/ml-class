#!/usr/bin/env python
# encoding: utf-8
"""
Object-orient back-propagation framework based on code by Prof. Yann LeCun

"""

from __future__ import division

__all__ = ['Machine']

import numpy as np
from modules import InputModule

class Machine(object):
    """
    A set of modules starting with an InputModule and ending with a Loss module

    The modules in the _modules member are stored in the following fashion:

                              --------
        self._modules[-1]:   |  LOSS  |<--- Y
                              --------   |
                                  ^      |
                               XN |      |
             ...                 ...     |
                                  ^      |
                               X1 |      |
                              --------   |
        self._modules[1]:    | MODULE |  |
                              --------   |
                                  ^      |
                               X0 |      |
                              --------   |
        self._modules[0]:    | INPUT  |--
                              --------

    Parameters
    ----------
    data : Dataset
        This is a dataset with the training and test samples.

    """
    def __init__(self, data):
        self._data = data
        self.x, self.y = data.training_set
        self._modules = [InputModule(self.x.shape[1], self.y.shape[1])]

    def add_module(self, module, args=()):
        self._modules.append(module(*args, prev_module=self._modules[-1]))

    def train_sample(self, sample, label, eta=0.00005, decay=0.1):
        self._modules[0].set_current_sample(sample, label)
        self._modules[-1].do_fprop()
        self._modules[0].do_bprop()

        for m in self._modules:
            if m.w is not None:
                m.w -= eta*(m.dw.T + decay*m.w)

    def train(self, eta=0.005, decay=0.0):
        for i in xrange(self._data.size_train):
            self.train_sample(self.x[i], self.y[i], eta=eta, decay=decay)


if __name__ == '__main__':
    pass

