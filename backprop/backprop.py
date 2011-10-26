#!/usr/bin/env python
# encoding: utf-8
"""
Object-orient back-propagation framework based on code by Prof. Yann LeCun

"""

from __future__ import division

__all__ = ['State', 'Parameter']

import numpy as np

class State(object):
    """
    Wrapper to carry variables between modules

    Parameters
    ----------
    shape : tuple
        The shape of the state vector

    """
    def __init__(self, shape):
        self.shape = shape
        self.x     = np.zeros(shape)
        self.dx    = np.zeros(shape)

class Parameter(object):
    """
    Wrapper around the weight vector for a module

    """
    def __init__(self):
        self._states = []

    def add_state(self, state):
        """
        Add a new state to the parameter set

        Parameters
        ----------
        state : State
            A State object that will be added to the stack

        """
        self._state.append(state)

if __name__ == '__main__':
    pass

