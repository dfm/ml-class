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
        self._modules = [InputModule(data.nvariables, data.nclass)]

    def add_module(self, module, args=(), kwargs={}):
        self._modules.append(module(*args, prev_module=self._modules[-1], **kwargs))

    def run(self, sample, label):
        self._modules[0].set_current_sample(sample, label)
        self._modules[-1].do_fprop()
        self._modules[0].do_bprop()

    def train_sample(self, sample, label, eta, decay):
        self.run(sample, label)

        for m in self._modules:
            if m.w is not None:
                m.w -= eta*(m.dw.T + decay*m.w)

    def training_sweep(self, eta, decay):
        x, y = self._data.training_set
        loss = 0.0
        for i in xrange(self._data.size_train):
            self.train_sample(x[i], y[i], eta, decay)
            loss += self._modules[-1].x
        return loss/self._data.size_train

    def train(self, maxiter=100, tol=5.25e-3, eta=0.01, decay=0.0001):
        loss0 = self.training_sweep(0.0, 0.0)

        for i in xrange(maxiter):
            loss = self.training_sweep(eta, decay)
            print loss

            # check for convergence
            if i > 5 and np.abs((loss-loss0)/loss) < tol:
                break
            loss0 = loss
        else:
            print "Warning: convergence criterion wasn't met after %d iterations"\
                    %maxiter

    def test_sample(self, sample, label):
        self.run(sample, np.ones(label.shape, dtype=int))
        losses = self._modules[-1].losses.flatten()
        error  = losses[label==1] != np.min(losses)
        return self._modules[-1].x, error[0]

    def test(self, training_set=False):
        if training_set:
            x, y = self._data.training_set
            N = self._data.size_train
        else:
            x, y = self._data.test_set
            N = self._data.size_test
        tot_loss, tot_err = 0., 0.
        for i in xrange(N):
            loss, error = self.test_sample(x[i], y[i])
            tot_loss += loss
            tot_err  += error
        return tot_loss/N, tot_err/N

