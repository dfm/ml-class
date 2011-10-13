#!/usr/bin/env python
# encoding: utf-8
"""
Linear classifiers implemented for Homework 1 in Yann LeCun's ML class @ NYU

All original code by Daniel Foreman-Mackey

"""

from __future__ import division

__all__ = ['LinearClassifier']

import numpy as np

from dataset import Dataset, LinearlySeperableDataset

class LinearClassifier(object):
    """
    The abstract base class for my Linear Classifier Machines

    """
    def __init__(self, dataset):
        self._dataset = dataset
        self._weights = np.ones(dataset.nvariables)
        self._bias = 0.0
        self._sum = 0.0

    def run(self, sample):
        self._sum = self._bias + np.dot(self._weights, sample)
        return 1 if self.loss(1) < self.loss(-1) else -1

    def loss(self):
        raise NotImplementedError()

    def learn_sample(self, sample, label):
        raise NotImplementedError()

    def train(self, niter=1):
        """
        Train the classifier on a dataset using stochastic gradient

        Parameters
        ----------
        niter : int, optional
            The number of iterations to sweep over the full training set
            (default: 1)

        """
        train_in, train_out = self._dataset.training_set
        for iteration in range(niter):
            for i in range(self._dataset.size_train):
                r = self.run(train_in[i])
                self.learn_sample(train_in[i], train_out[i])

    def test_sample(self, sample, label):
        """
        Test the prediction on a particular sample

        Parameters
        ----------
        sample : numpy.ndarray
            The test sample

        label : int
            The true label

        Returns
        -------
        loss : float
            The loss provided by this sample

        error : bool
            Is this prediction wrong?

        """
        r = self.run(sample)
        return (self.loss(label),
                ((r > 0 and label <= 0) or (r <= 0 and label > 0)))

    def test(self, on_training_set=False, on_full_set=False):
        """
        Test the prediction power of the optimized machine

        Parameters
        ----------
        on_training_set : bool, optional
            Return the test results calculated on the training set instead of
            the test set (default: False)

        Returns
        -------
        average_loss : float
            The loss function averaged over test samples

        fractional_error : float
            The fractional error on the test samples

        """
        errors, total_loss = 0, 0.0
        if on_full_set:
            test_in, test_out = self._dataset._inputs, self._dataset._outputs
        elif on_training_set:
            test_in, test_out = self._dataset.training_set
        else:
            test_in, test_out = self._dataset.test_set
        n = len(test_out)
        for i in range(n):
            loss, error = self.test_sample(test_in[i], test_out[i])
            errors += error
            total_loss += loss

        return total_loss/n, errors/n

    def stats(self):
        stats = {}
        stats['train-loss'], stats['train-error'] = self.test(on_training_set=True)
        stats['test-loss'], stats['test-error'] = self.test()
        stats['full-loss'], stats['full-error'] = self.test(on_full_set=True)
        return stats

    def __str__(self):
        s =  "TRAIN: loss = %7.4f, error = %6.4f\n"%self.test(on_training_set=True)
        s += " TEST: loss = %7.4f, error = %6.4f"%self.test()
        return s

class Perceptron(LinearClassifier):
    def __init__(self, *args, **kwargs):
        super(Perceptron, self).__init__(*args, **kwargs)

    def loss(self, label):
        return (np.sign(self._sum) - label) * self._sum

    def learn_sample(self, sample, label, eta=0.003):
        delta = eta * (label - np.sign(self._sum))
        self._weights += delta * sample
        self._bias    += delta

class LinearRegression(LinearClassifier):
    def __init__(self, *args, **kwargs):
        super(LinearRegression, self).__init__(*args, **kwargs)

    def loss(self, label):
        return 0.5 * (self._sum - label)**2

    def learn_sample(self, sample, label, eta=0.003):
        delta = eta * (label - np.sign(self._sum))
        self._weights += delta * sample
        self._bias    += delta

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tests',
            help='Run some tests on a mock dataset',
            action='store_true')
    parser.add_argument('-p', '--perceptron',
            help='Train/test the perceptron algorithm on the spambase dataset',
            action='store_true')
    parser.add_argument('-l', '--linear',
            help='Train/test linear regression on the spambase dataset',
            action='store_true')
    args = parser.parse_args()

    if args.tests:
        import matplotlib.pyplot as pl

        data = LinearlySeperableDataset(ndim=2, nsamples=4000)
        machine = Perceptron(data)
        for i in range(20):
            machine.train()

        inds = data._outputs == 1
        pl.plot(data._inputs[inds,0], data._inputs[inds,1], '.g')
        pl.plot(data._inputs[~inds,0], data._inputs[~inds,1], '.r')

        w0 = data.weights
        w  = machine._weights
        x = np.linspace(-0.5, 0.5, 100)
        y0 = -(w0[0]*x+data.bias)/w0[1]
        y  = -(w[0]*x+machine._bias)/w[1]

        pl.plot(x, y0, 'k', lw=3.0, alpha=0.5)
        pl.plot(x, y, 'b', lw=2.0)
        pl.xlim([-0.5, 0.5])
        pl.ylim([-0.5, 0.5])

        loss,error = machine.test()
        pl.title("loss = %.4f, error = %.4f"%(loss,error))

        pl.savefig('perceptron_test.png')

    # table layout strings
    header = "Iter%11s%11s%11s%11s%11s%11s"%\
                ("tr-loss","tr-error","test-loss", "test-error", "full-loss",
                        "full-error")
    layout  = "%(train-loss)10.4f %(train-error)10.4f "
    layout += "%(test-loss)10.4f %(test-error)10.4f "
    layout += "%(full-loss)10.4f %(full-error)10.4f"

    if args.perceptron:
        data = Dataset('spambase.data', train=500, test=1000,
                shuffle=True, normalize=True, binary=True)

        machine = Perceptron(data)

        print "Perceptron"
        print "=========="
        print header
        print "   0",
        print layout%machine.stats()
        for i in range(20):
            machine.train()
            print "%4d"%(i+1),
            print layout%machine.stats()
        print

    if args.linear:
        data = Dataset('spambase.data', train=500, test=1000,
                shuffle=True, normalize=True, binary=False)
        machine = LinearRegression(data)

        print "Linear Regression"
        print "====== =========="
        print header
        print "   0",
        print layout%machine.stats()
        for i in range(20):
            machine.train()
            print "%4d"%(i+1),
            print layout%machine.stats()



