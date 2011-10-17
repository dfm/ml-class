#!/usr/bin/env python
# encoding: utf-8
"""
My homework...

"""

from __future__ import division

__all__ = ['hw1']

import argparse

import numpy as np

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
    parser.add_argument('--hyperparams',
            help='Test a grid in eta',
            action='store_true')
    parser.add_argument('-s', '--size',
            help='Test a grid in training set size',
            action='store_true')
    parser.add_argument('--alpha',
            help='Test a grid in alpha',
            action='store_true')
    parser.add_argument('--beta',
            help='Test a grid in beta',
            action='store_true')
    parser.add_argument('-v', '--verbose',
            help='Will the machine spit out all the results at each step?',
            action='store_true')
    args = parser.parse_args()

    # check for stupidness
    assert(not args.test or not args.hyperparams)
    if args.hyperparams:
        assert(args.linear != args.logistic)

    if args.size:
        f = open('ntrain.dat', 'w')
        
        for ntrain in [10, 30, 100, 500, 3000]:
            data = Dataset('dataset/spambase.data', train=ntrain, test=1000,
                shuffle=True, normalize=True)
            machine = LogisticRegression(data, eta=0.002)
            stats, i = machine.train(verbose=False, maxiter=100)
            f.write("%d %d "%(ntrain, i))
            f.write("%(train-loss)e %(train-error)e "%stats)
            f.write("%(test-loss)e %(test-error)e\n"%stats)

        f.close()

        return

    if args.alpha:
        f = open('alpha.dat', 'w')
        
        for ntrain in [10, 30, 100]:
            data = Dataset('dataset/spambase.data', train=ntrain, test=1000,
                shuffle=True, normalize=True)
            for alpha in np.linspace(0, 1, 50):
                machine = LogisticRegression(data, eta=0.002, alpha=alpha)
                stats, i = machine.train(verbose=False, maxiter=100)
                f.write("%d %f %d "%(ntrain, alpha, i))
                f.write("%(train-loss)e %(train-error)e "%stats)
                f.write("%(test-loss)e %(test-error)e\n"%stats)

        f.close()

        return

    if args.beta:
        f = open('beta.dat', 'w')
        
        for ntrain in [10, 30, 100]:
            data = Dataset('dataset/spambase.data', train=ntrain, test=1000,
                    shuffle=True, normalize=True)
            
            for beta in np.linspace(0, 1, 100):
                machine = LogisticRegression(data, eta=0.002, beta=beta)
                stats, i = machine.train(verbose=False, maxiter=200)
                f.write("%d %f %d "%(ntrain, beta, i))
                f.write("%(train-loss)e %(train-error)e "%stats)
                f.write("%(test-loss)e %(test-error)e\n"%stats)

        f.close()

        return

    # Run some tests on a truely linearly seperable system
    if args.test:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as pl

        data = LinearlySeperableDataset(ndim=2, nsamples=4000, train=int(args.ntrain),
                test=1000)
    else:
        # Run tests on "spambase" dataset
        data = Dataset('dataset/spambase.data', train=int(args.ntrain), test=1000,
                shuffle=True, normalize=True)

    if args.hyperparams:
        f = open('hyperparams.dat', 'w')
        for log10eta in np.linspace(-3.5, -1.5, 50):
            eta = 10**log10eta
            f.write("%f "%(eta))
            if args.linear:
                machine = LinearRegression(data, eta=eta)
            if args.logistic:
                machine = LogisticRegression(data, eta=eta)
            stats, i = machine.train(verbose=args.verbose)
            f.write("%d "%i)
            [f.write("%e "%stats[k]) for k in \
                    ['train-loss', 'train-error', 'test-loss', 'test-error']]
            f.write("\n")
        f.close()
        return

    if args.perceptron:
        machine = Perceptron(data)
        machine.train(verbose=args.verbose)

    if args.linear:
        machine = LinearRegression(data, eta=0.00003)
        machine.train(verbose=args.verbose)

    if args.direct:
        machine = LinearRegression(data)
        machine.solve()

    if args.logistic:
        machine = LogisticRegression(data, eta=0.002)
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

