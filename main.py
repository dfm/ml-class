#!/usr/bin/env python
# encoding: utf-8
"""
Experiments for Homework #2 in in Yann LeCun's Machine Learning class a NYU

All original code by Daniel Foreman-Mackey
Most of the skeleton code has been loosely "ported" from the provided LUSH code.

"""

import numpy as np

from backprop import Machine, modules
from dataset import Dataset

data = Dataset('dataset/isolet1+2+3+4.data', test_fn='dataset/isolet5.data')

def single_layer_test():
    mach = Machine(data)
    mach.add_module(modules.LinearModule, kwargs={'dim_out': data.nclass})
    mach.add_module(modules.BiasModule)
    mach.add_module(modules.SigmoidModule)
    mach.add_module(modules.EuclideanModule)

    print mach.test()
    for i in range(100):
        mach.train()
        print mach.test(), mach.test(training_set=True)

if __name__ == '__main__':
    single_layer_test()

