#!/usr/bin/env python
# encoding: utf-8
"""
Experiments for Homework #2 in in Yann LeCun's Machine Learning class a NYU

All original code by Daniel Foreman-Mackey
Most of the skeleton code has been loosely "ported" from the provided LUSH code.

"""

import sys

import numpy as np

from backprop import Machine, modules
from dataset import Dataset

data = None
# data = Dataset('dataset/spambase.data')

def run_machine(mach, save=True, **kwargs):
    N = mach.nparams
    print "Running machine w/ architecture"
    print "\t", mach
    print "and %d parameters"%(N)
    print "Initial L = %.4f, f_error = %.4f on training set"%(mach.test(training_set=True))
    print "Initial L = %.4f, f_error = %.4f on test set"%(mach.test())
    mach.train(**kwargs)
    print "Final L = %.4f, f_error = %.4f on training set"%(mach.test(training_set=True))
    testing = mach.test()
    print "Final L = %.4f, f_error = %.4f on test set\n"%(testing)
    if save:
        f = open('results.dat', 'a')
        f.write("%d %.4f\n"%(N,testing[1]))
        f.close()

def logistic_regression():
    mach = Machine(data)

    mach.add_module(modules.LinearModule, kwargs={'dim_out': data.nclass})
    mach.add_module(modules.BiasModule)
    mach.add_module(modules.SigmoidModule)
    mach.add_module(modules.EuclideanModule)

    run_machine(mach, eta=0.0025, decay=0.0025, tol=5.25e-5)

def single_layer():
    mach = Machine(data)

    mach.add_module(modules.LinearModule, kwargs={'dim_out': data.nclass})
    mach.add_module(modules.BiasModule)
    mach.add_module(modules.SoftMaxModule)
    mach.add_module(modules.CrossEntropyModule)

    run_machine(mach, eta=0.001, decay=0.0001, tol=1.25e-2)

def double_layer(nhidden=80, eta=0.001, decay=0.0001, tol=1.25e-2):
    mach = Machine(data)

    mach.add_module(modules.LinearModule, kwargs={'dim_out': nhidden})
    mach.add_module(modules.BiasModule)
    mach.add_module(modules.SigmoidModule)

    mach.add_module(modules.LinearModule, kwargs={'dim_out': data.nclass})
    mach.add_module(modules.BiasModule)
    mach.add_module(modules.SoftMaxModule)
    mach.add_module(modules.CrossEntropyModule)

    run_machine(mach, eta=eta, decay=decay, tol=tol)

def triple_layer(nhidden=80, eta=0.001, decay=0.0001, tol=1.25e-2):
    mach = Machine(data)

    mach.add_module(modules.LinearModule, kwargs={'dim_out': nhidden})
    mach.add_module(modules.BiasModule)
    mach.add_module(modules.SigmoidModule)

    mach.add_module(modules.LinearModule, kwargs={'dim_out': nhidden})
    mach.add_module(modules.BiasModule)
    mach.add_module(modules.SigmoidModule)

    mach.add_module(modules.LinearModule, kwargs={'dim_out': data.nclass})
    mach.add_module(modules.BiasModule)
    mach.add_module(modules.SoftMaxModule)
    mach.add_module(modules.CrossEntropyModule)

    run_machine(mach, eta=eta, decay=decay, tol=tol)

def rbf_hybrid(nhidden=80, eta=0.001, decay=0.0001, tol=1.25e-2):
    templates = np.random.randn(nhidden*data.nclass).reshape(nhidden, data.nclass)

    mach = Machine(data)

    mach.add_module(modules.LinearModule, kwargs={'dim_out': nhidden})
    mach.add_module(modules.BiasModule)
    mach.add_module(modules.SigmoidModule)

    mach.add_module(modules.RBFModule, args=[templates])
    mach.add_module(modules.BiasModule)
    mach.add_module(modules.SoftMaxModule)
    mach.add_module(modules.CrossEntropyModule)

    run_machine(mach, save=False, eta=eta, decay=decay, tol=tol)

def svm(nhidden=30, eta=0.001, decay=0.001, tol=1.25e-4):
    templates = data.training_set[0][np.random.randint(data.size_train, size=nhidden),:].T
    mach = Machine(data)
    mach.add_module(modules.RBFModule, args=[templates])
    mach.add_module(modules.SoftMaxModule)
    mach.add_module(modules.LinearModule, kwargs={'dim_out': data.nclass})
    mach.add_module(modules.BiasModule)
    mach.add_module(modules.SoftMaxModule)
    mach.add_module(modules.CrossEntropyModule)

    run_machine(mach, save=False, eta=eta, decay=decay, tol=tol)

if __name__ == '__main__':
    if '--optional' in sys.argv:
        data = Dataset('dataset/isolet1+2+3+4.data', test_fn='dataset/isolet5.data',
            train=4000, test=1000)
        triple_layer(nhidden=40)
        rbf_hybrid()
    else:
        N = 1
        if len(sys.argv) > 1:
            N = int(sys.argv[1])

        for i in range(N):
            print "trial %d"%i
            data = Dataset('dataset/isolet1+2+3+4.data', test_fn='dataset/isolet5.data',
                train=4000, test=1000)

            logistic_regression()
            single_layer()
            double_layer(nhidden=10)
            double_layer(nhidden=20)
            double_layer(nhidden=40)
            double_layer(nhidden=80)

