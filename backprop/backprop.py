#!/usr/bin/env python
# encoding: utf-8
"""
Object-orient back-propagation framework based on code by Prof. Yann LeCun

"""

from __future__ import division

__all__ = ['Machine']

import os
import sys

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

    def __str__(self):
        r = ">".join([str(m) for m in self._modules])
        return r

    def add_module(self, module, args=(), kwargs={}):
        self._modules.append(module(*args, prev_module=self._modules[-1], **kwargs))

    def train_sample(self, sample, label, eta, decay):
        self._modules[0].set_current_sample(sample, label)
        self._modules[-1].do_fprop()
        self._modules[0].do_bprop()

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

    def train(self, maxiter=100, miniter=5, tol=5.25e-5, eta=0.1, decay=0.0001):
        loss0 = 0.0
        msg = ""
        for i in xrange(maxiter):
            loss = self.training_sweep(eta, decay)
            sys.stdout.write("\b"*len(msg))
            msg = "."*i + " L = %.4e"%(loss)
            sys.stdout.write(msg)
            sys.stdout.flush()

            # check for convergence
            diff = np.abs((loss-loss0)/loss)
            if i > miniter and diff < tol:
                print
                print "converged after %d iterations"%i
                break
            loss0 = loss
        else:
            print
            print "Warning: convergence criterion (%.4e < %.4e)"%(diff,tol)+\
                    " wasn't met after %d iterations"\
                    %maxiter

    def test_sample(self, sample, label):
        # try to classify the sample
        self._modules[0].set_current_sample(sample, np.ones(label.shape, dtype=int))
        self._modules[-1].do_fprop()

        losses = self._modules[-1].losses.flatten()
        error  = losses[label==1] != np.min(losses)

        # now get the loss given the _correct_ classification
        self._modules[0].set_current_sample(sample, label)
        self._modules[-1].do_fprop()

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

    @property
    def nparams(self):
        n = 0
        for m in self._modules:
            if m.w is not None:
                n += np.prod(m.w.shape)
        return n

