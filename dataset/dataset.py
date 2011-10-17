#!/usr/bin/env python
# encoding: utf-8
"""
Dataset interface module

Based on dataset.lsh provided by Prof. LeCun

"""

from __future__ import division

__all__ = ['Dataset', 'LinearlySeperableDataset']

import numpy as np

class Dataset(object):
    """
    Object for wrapping a ML dataset

    Parameters
    ----------
    fn : str, optional
        The path to the CSV data file. Provide either this or the data directly

    data : numpy.ndarray, optional
        The raw data. Either this or `fn` must not be None.

    outcol : int, optional
        The column that the output values will be stored in (default: -1)

    train : int, optional
        The number of samples to use as a training set (default: 1000)

    test : int, optional
        The number of samples to use as a test set (default: everything that's
        not in the training set). Note: train+test must be in the interval (0,N]

    shuffle : bool, optional
        Should we shuffle the samples (default: True)

    normalize : bool, optional
        Should we normalize the data to zero mean and unit variance (default: True)

    binary : bool, optional
        If the output values are binary, convert the zeros to -1 (default: True)

    Usage
    -----
    >>> data = Dataset('spambase.data', train=500)
    >>> in_train, out_train = data.training_set
    >>> in_test, out_test = data.test_set

    """
    def __init__(self, fn=None, data=None, outcol=-1, train=1000, test=None,
            shuffle=True, normalize=True, binary=True):
        assert fn is not None or data is not None

        if fn is not None:
            data = np.array([line.split(',') for line in open(fn)], dtype=float)

        # dimensions
        self.size = data.shape[0]
        self.nvariables = data.shape[-1] - 1
        self.size_train = int(train)
        if test is None:
            self.size_test = self.size-self.size_train
        else:
            self.size_test = int(test)

        assert (test is None and 0 < int(train) < self.size) \
                or (test is not None and 0 < int(self.size_test+train) <= self.size)

        if shuffle:
            np.random.shuffle(data)

        # slice output vector out of data file
        if outcol < 0:
            outcol += data.shape[-1]
        inds = np.arange(data.shape[-1]) != outcol
        self._inputs  = data[:,inds]
        self._outputs = data[:,outcol]
        if binary:
            self._outputs[self._outputs == 0] = -1

        # normalize and split the dataset into training and test sets
        if normalize:
            self.normalize() # normalize performs the split itself
        else:
            # we should split the dataset if we didn't already normalize
            self._split()

    def _split(self):
        # split the dataset into training and test sets
        self._inputs_train  = self._inputs[:self.size_train]
        self._outputs_train = self._outputs[:self.size_train]
        delta = self.size_train + self.size_test
        self._inputs_test   = self._inputs[self.size_train:delta]
        self._outputs_test  = self._outputs[self.size_train:delta]

    @property
    def training_set(self):
        return (self._inputs_train, self._outputs_train)

    @property
    def test_set(self):
        return (self._inputs_test, self._outputs_test)

    def normalize(self):
        """
        Normalize the dataset to have zero mean and unit variance

        """
        self._inputs -= np.mean(self._inputs, axis=0)
        self._inputs /= np.var(self._inputs, axis=0)
        self._split()

class LinearlySeperableDataset(Dataset):
    """
    Generate a truly linearly separable dataset

    Parameters
    ----------
    ndim : int, optional
        The number of dimensions (default: 20)

    nsamples : int, optional
        The number of samples to generate (default: 4000)

    """
    def __init__(self, ndim = 20, nsamples = 4000, **kwargs):
        out_data = np.ones(nsamples)
        while not np.abs(np.sum(out_data)) < nsamples/2:
            data = 0.5-np.random.rand(ndim*nsamples).reshape((nsamples, ndim))
            self.weights = 0.5-np.random.rand(ndim)
            self.bias = 0.5-np.random.rand()
            out_data = np.sign(np.dot(self.weights, data.T) + self.bias)
        data = np.concatenate((data, np.atleast_2d(out_data).T), axis=-1)
        kwargs['binary'] = False
        kwargs['normalize'] = False
        super(LinearlySeperableDataset, self).__init__(data=data, **kwargs)

    def __str__(self):
        return """Weights:
    %s
Bias: %f
"""%(str(self.weights), self.bias)

if __name__ == '__main__':
    import matplotlib.pyplot as pl
    dataset = LinearlySeperableDataset(ndim=2)
    inds = dataset._outputs == 1
    pl.plot(dataset._inputs[inds,0], dataset._inputs[inds,1], '.g')
    pl.plot(dataset._inputs[~inds,0], dataset._inputs[~inds,1], '.r')

    w = dataset.weights
    x = np.linspace(-0.5, 0.5, 100)
    y = -(w[0]*x+dataset.bias)/w[1]

    pl.plot(x,y,'b')

    pl.show()

