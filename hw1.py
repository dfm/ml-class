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

def hw1():
    # Start by analysing the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--ntrain', help="Number of training samples",
            default=1000)
    parser.add_argument('-t', '--test',
            help='Run some tests on a mock dataset',
            action='store_true')
    parser.add_argument('-p', '--perceptron',
            help='Train/test the perceptron algorithm on the dataset/spambase.dataset',
            action='store_true')
    parser.add_argument('-l', '--linear',
            help='Train/test linear regression on the dataset/spambase.dataset',
            action='store_true')
    parser.add_argument('-d', '--direct',
            help='Direct solution of linear regression',
            action='store_true')
    parser.add_argument('-g', '--logistic',
            help='Train/test logistic regression on the dataset/spambase.dataset',
            action='store_true')
    parser.add_argument('-v', '--verbose',
            help='Will the machine spit out all the results at each step?',
            action='store_true')
    args = parser.parse_args()

    # Run some tests on a truely linearly seperable system
    if args.test:
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as pl

        data = LinearlySeperableDataset(ndim=2, nsamples=4000, train=int(args.ntrain),
                test=1000)
    else:
        # Run tests on "spambase" dataset
        data = Dataset('dataset/spambase.data', train=int(args.ntrain), test=1000,
                shuffle=True, normalize=True)

    if args.perceptron:
        machine = Perceptron(data)
        machine.train(verbose=args.verbose)

    if args.linear:
        machine = LinearRegression(data, eta=0.00003, alpha=0.0)
        machine.train(verbose=args.verbose)

    if args.direct:
        machine = LinearRegression(data)
        machine.solve()

    if args.logistic:
        machine = LogisticRegression(data, eta=0.01, alpha=0.1)
        machine.train(verbose=args.verbose)

    if args.test:
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

if __name__ == '__main__':
    hw1()

