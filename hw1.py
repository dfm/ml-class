#!/usr/bin/env python
# encoding: utf-8
"""
My homework...

"""

from __future__ import division

__all__ = ['hw1']

import argparse

from dataset import Dataset, LinearlySeperableDataset
from linear import Perceptron, LinearRegression, LogisticRegression

# Define a table layout
header = "Iter%11s%11s%11s%11s%11s%11s"%\
            ("tr-loss","tr-error","test-loss", "test-error", "full-loss",
                    "full-error")
layout  = "%(train-loss)10.4f %(train-error)10.4f "
layout += "%(test-loss)10.4f %(test-error)10.4f "
layout += "%(full-loss)10.4f %(full-error)10.4f"

def hw1():
    # Start by analysing the command line arguments
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
    parser.add_argument('-g', '--logistic',
        help='Train/test logistic regression on the spambase dataset',
            action='store_true')
    args = parser.parse_args()

    # Run some tests on a truely linearly seperable system
    if args.tests:
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as pl

        data = LinearlySeperableDataset(ndim=2, nsamples=4000, train=300,
                binary=True)
        machine = LogisticRegression(data, eta=0.01, alpha=0.1)
        for i in range(10):
            machine.train()
            print machine._weights, machine.test()

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

        pl.savefig('test.png')

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
                shuffle=True, normalize=True, binary=True)
        machine = LinearRegression(data, eta=0.0002, alpha=0.05)
        machine.solve()

        print "Linear Regression"
        print "====== =========="
        print header
        print "   0",
        print layout%machine.stats()
        for i in range(20):
            machine.train()
            print "%4d"%(i+1),
            print layout%machine.stats()

    if args.logistic:
        data = Dataset('spambase.data', train=10, test=1000,
                shuffle=True, normalize=True, binary=True)
        machine = LogisticRegression(data, eta=0.01, alpha=0.1)

        print "Logistic Regression"
        print "======== =========="
        print header
        print "   0",
        print layout%machine.stats()
        for i in range(20):
            machine.train()
            print "%4d"%(i+1),
            print layout%machine.stats()

if __name__ == '__main__':
    hw1()

