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

    """
    def __init__(self, data):
        self._modules = [InputModule(*(data.training_set))]

    def add_module(self, module, args=()):
        self._modules.append(module(*args, prev_module=self._modules[-1]))

    def add_loss_module(self, loss_module):
        self._modules.append(loss_module(self._modules[0].y,
            prev_module=self._modules[-1]))

    def update(self, eta=0.05, decay=0.1):
        self._modules[-1].do_fprop()
        self._modules[0].do_bprop()

        for m in self._modules:
            if m.w is not None:
                m.w -= eta*(m.dw + decay*m.w)

if __name__ == '__main__':
    pass

