#!/usr/bin/env python
# encoding: utf-8
"""
Generate some sick plots.

"""

from __future__ import division

__all__ = ['']

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl

def make_plots():
    ax = pl.figure().add_subplot(111)
    data = np.array([line.split() for line in open('results.dat')], dtype=float)

    types = {'logistic-regression': [16068], 'single-layer': [16069],
            'double-layer': [6467,12907,25787,51547]}
    s = ['.k', '.b', '.g']

    for i, t in enumerate(list(types)):
        tmp = np.empty((0,data.shape[-1]))
        for n in types[t]:
            tmp = np.vstack([tmp, data[data[:,0] == n, :]])
        ax.plot(tmp[:,0], tmp[:,1], s[i], label=t)

    pl.legend()
    ax.axhline(0.05)
    ax.set_xlabel(r'$N_W$',fontsize=16)
    ax.set_ylabel(r'$f_\mathrm{err}$', fontsize=16)
    pl.savefig('results.png')

    ax.set_ylim([0,0.15])
    pl.savefig('results_scaled.png')

if __name__ == '__main__':
    make_plots()

