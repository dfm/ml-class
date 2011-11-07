#!/usr/bin/env python
# encoding: utf-8
"""
Object-orient back-propagation framework based on code by Prof. Yann LeCun

"""

from __future__ import division

__all__ = ['Machine']

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
        """
        Instatiate and append a new module to the machine

        Parameters
        ----------
        module : LearningModule
            This should be the class not an instance.

        args : tuple
            The arguments for module's constructor

        kwargs : dict
            The keyword arguments for module's constructor

        """
        self._modules.append(module(*args, prev_module=self._modules[-1], **kwargs))

    def train_sample(self, sample, label, eta, decay):
        """
        Run a single stochastic gradient update for a given training sample

        Parameters
        ----------
        sample : numpy.ndarray
            The input vector

        label : numpy.ndarray
            The target output

        eta : float
            The learning rate

        decay : float
            The decay rate

        """
        self._modules[0].set_current_sample(sample, label)
        self._modules[-1].do_fprop()
        self._modules[0].do_bprop()

        for m in self._modules:
            if m.w is not None:
                m.w -= eta*(m.dw.T + decay*m.w)

    def training_sweep(self, eta, decay):
        """
        Perform a single stochastic gradient sweep over the full training set

        Parameters
        ----------
        eta : float
            The learning rate

        decay : float
            The decay rate

        Returns
        -------
        loss : float
            The mean loss over training samples

        """
        x, y = self._data.training_set
        loss = 0.0
        for i in xrange(self._data.size_train):
            self.train_sample(x[i], y[i], eta, decay)
            loss += self._modules[-1].x
        return loss/self._data.size_train

    def train(self, maxiter=100, miniter=5, tol=5.25e-5, eta=0.1, decay=0.0001):
        """
        Train the machine

        Parameters
        ----------
        maxiter : int, optional
            Maximum number of stochastic gradient sweeps (default: 100)

        miniter : int, optional
            Set a minimum number of sweeps to avoid premature convergence
            (default: 5)

        tol : float, optional
            The convergence criterion for the relative change in the loss function
            between training sweeps (default: 5.25e-5)

        eta : float, optional
            The learning rate (default: 0.1)

        decay : float, optional
            The decay rate (default: 0.0001)

        """
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
        """
        Test the classification of a given sample

        Parameters
        ----------
        sample : numpy.ndarray
            The input vector

        label : numpy.ndarray
            The target output

        """
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
        """
        Test the performance of the machine

        Parameters
        ----------
        training_set : bool, optional
            If True, test on the training set, otherwise, test on the test set
            (default: False)

        Returns
        -------
        loss : float
            The mean loss over all samples in the dataset

        err : float
            The fractional error in the classification

        """
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
        """
        Return the total number of parameters in this machine

        Returns
        -------
        nparams : int
            The total number of parameters in the machine

        """
        n = 0
        for m in self._modules:
            if m.w is not None:
                n += np.prod(m.w.shape)
        return n

