#!/usr/bin/env python
# encoding: utf-8
"""
Modules to use to construct a learning machine

"""

from __future__ import division

__all__ = ['LearningModule', 'InputModule', 'LinearModule', 'EuclideanModule',
        'BiasModule', 'SigmoidModule']

import numpy as np
import numpy.linalg

class LearningModule(object):
    """
    The abstract base class for a module

    Parameters
    ----------
    prev_modules : LearningModule
        The module that connects to this one

    """
    def __init__(self, prev_module=None, dim_out=None, dim_in=None):
        self.prev_module = prev_module
        self.next_module = None

        if prev_module is not None:
            # the input dimensions will be set by the previous module's output dimensions
            self.dim_in = self.prev_module.dim_out
            if dim_in is not None:
                assert(dim_in == self.dim_in)
        else:
            self.dim_in = dim_in

        # output dimensions default to the same as dim_in
        if dim_out is not None:
            self.dim_out = dim_out
        else:
            self.dim_out = self.dim_in

        if prev_module is not None:
            # connect the previous module
            self.prev_module.connect_module(self)

        self.w  = None                        # W
        self.dw = None                        # dE/dW
        self.x  = np.zeros((self.dim_out, 1)) # X_out
        self.dx = np.zeros((self.dim_in, 1))  # dE/dXin

        self.randomize()

    def randomize(self, **kwargs):
        pass

    def connect_module(self, next_module):
        assert(self.dim_out == next_module.dim_in)
        self.next_module = next_module

    def do_fprop(self):
        self.prev_module.do_fprop()
        self.y = self.prev_module.y
        self.fprop()

    def do_bprop(self):
        self.next_module.do_bprop()
        self.dy = self.next_module.dy
        self.bprop()
        assert (self.w is None and self.dw is None) or self.dw.T.shape == self.w.shape, repr(self)
        assert self.dx.shape == (1, self.dim_in), repr(self)

    def fprop(self):
        pass

    def bprop(self):
        pass

class InputModule(LearningModule):
    def __init__(self, dim_x, dim_y):
        self.dim_out = dim_x
        self.dim_y   = dim_y
        self.w, self.dw = None, None

    # override default fprop and bprop behaviour
    def do_fprop(self):
        pass

    def do_bprop(self):
        self.next_module.do_bprop()

    def current_sample(self):
        return self.x, self.y

    def set_current_sample(self, x, y):
        # NOTE: this only works when x and y are 0 or 1-D arrays (not 2-D)
        #       this converts the inputs to column vectors
        self.x, self.y = np.atleast_2d(x).T, np.atleast_2d(y).T

class EuclideanModule(LearningModule):
    def fprop(self):
        assert(self.prev_module.x.shape == self.prev_module.y.shape)
        self.losses = 0.5*(self.prev_module.x-self.prev_module.y)**2
        self.x = np.sum(self.losses)

    def do_bprop(self):
        self.dx = (self.prev_module.x - self.y).T
        self.dy = -self.dx

class LinearModule(LearningModule):
    def __init__(self, *args, **kwargs):
        super(LinearModule, self).__init__(*args, **kwargs)
        self.x = np.zeros((self.dim_out, 1))

    def randomize(self, **kwargs):
        kz = kwargs.pop('k', 1.0)/np.sqrt(self.dim_in)
        self.shape = (self.dim_out, self.dim_in)
        self.w  = (2*np.random.rand(*(self.shape)) - 1) * kz

    def fprop(self):
        self.x = np.dot(self.w, self.prev_module.x)
        assert(self.x.shape[0] == self.dim_out and self.x.shape[1] == 1)

    def bprop(self):
        self.dw = np.dot(self.next_module.dx.T, self.prev_module.x.T).T
        self.dx = np.dot(self.w.T, self.next_module.dx.T).T
        assert(self.dw.shape == self.shape[::-1])
        assert(self.dx.shape[1] == self.dim_in and self.dx.shape[0] == 1)

class BiasModule(LearningModule):
    def randomize(self, **kwargs):
        self.w = np.random.rand(self.dim_in, 1)

    def fprop(self):
        assert(self.prev_module.x.shape == self.w.shape)
        self.x = self.prev_module.x + self.w

    def bprop(self):
        self.dx = self.next_module.dx
        self.dw = self.next_module.dx

class SigmoidModule(LearningModule):
    def fprop(self):
        self.x = np.tanh(self.prev_module.x)

    def bprop(self):
        # dtanh = 1/cosh^2
        # careful with dimensions
        self.dx = self.next_module.dx/np.cosh(self.prev_module.x.T)**2
        assert(self.dx.shape == self.next_module.dx.shape)

class SoftMax(LearningModule):
    def randomize(self, **kwargs):
        self.w = np.random.rand()

    def fprop(self):
        self.x = np.exp(-self.w * self.prev_module.x)
        self.x /= np.sum(self.x)

    def bprop(self):
        # (dX_out/dX_in)_k = beta * ( (X_{out,k})^2 - X_{out,k} )
        # (dX_out/dbeta)_k = X_{in,k} . ( (X_{out,k})^2 - X_{out,k} )
        delta = self.x**2-self.x
        self.dx = self.w * delta
        self.dw = np.dot( self.prev_module.x, delta )

