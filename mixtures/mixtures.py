#!/usr/bin/env python
# encoding: utf-8
"""
Gaussian mixture models

"""

from __future__ import division

__all__ = ['MixtureModel', 'KMeansConvergenceError']

import numpy as np

import _algorithms

class KMeansConvergenceError(Exception):
    pass

class MixtureModel(object):
    """
    Gaussian mixture model

    P data points in D dimensions with K clusters

    Shapes
    ------
    data -> (P, D)
    means -> (D, K)

    """
    def __init__(self, K, data):
        self._K    = K
        self._data = np.atleast_2d(data)
        self._lu   = None

        self._means = data[np.random.randint(data.shape[0],size=self._K),:].T
        self._cov   = [np.cov(data,rowvar=0)]*self._K
        self._as    = np.random.rand(K)
        self._as /= np.sum(self._as)

    @property
    def K(self):
        return self._K

    @K.setter
    def set_K(self, k):
        self._K = k

    @property
    def means(self):
        return self._means.T

    # ================= #
    # K-Means Algorithm #
    # ================= #

    def run_kmeans(self, maxiter=200, tol=1e-8, verbose=True):
        means = self.means
        self._kmeans_rs = np.zeros(self._data.shape[0], dtype=int)
        _algorithms.kmeans(self._data, means, self._kmeans_rs, tol, maxiter)
        self._means = means.T

    # ============ #
    # EM Algorithm #
    # ============ #

    def run_em(self, maxiter=400, tol=1e-8, verbose=True):
        """
        Fit the given data using EM

        """
        L = None
        for i in xrange(maxiter):
            newL = self._expectation()
            self._maximization()
            if L is None:
                L = newL
            else:
                dL = np.abs(newL-L)
                if dL < tol:
                    break
                L = newL
        if i < maxiter-1:
            if verbose:
                print "EM converged after %d iterations"%(i)
        else:
            print "Warning: EM didn't converge after %d iterations"%(i)

    def _multi_gauss(self, k, X):
        # X.shape == (P,D)
        # self._means.shape == (D,K)
        # self.cov[k].shape == (D,D)
        det = np.linalg.det(self._cov[k])

        # X1.shape == (P,D)
        X1 = X - self._means[None,:,k]

        # X2.shape == (P,D)
        X2 = np.linalg.solve(self._cov[k], X1.T).T

        p = -0.5*np.sum(X1 * X2, axis=1)

        return 1/np.sqrt( (2*np.pi)**(X.shape[1]) * det )*np.exp(p)

    def _expectation(self):
        # self._rs.shape == (P,K)
        L, self._rs = self._calc_prob(self._data)
        return np.sum(L, axis=0)

    def _maximization(self):
        # Nk.shape == (K,)
        Nk = np.sum(self._rs, axis=0)
        # self._means.shape == (D,K)
        self._means = np.sum(self._rs[:,None,:] * self._data[:,:,None], axis=0)
        self._means /= Nk[None,:]
        self._cov = []
        Cprior = np.cov(data,rowvar=0)
        for k in range(self._K):
            # D.shape == (P,D)
            D = self._data - self._means[None,:,k]
            # FIXME: bogus crap
            self._cov.append(\
                    (np.dot(D.T, self._rs[:,k,None]*D) \
                    + Cprior)/(Nk[k]+1) \
                    )
            # self._cov.append(np.dot(D.T, self._rs[:,k,None]*D)/Nk[k])
        self._as = Nk/self._data.shape[0]

    def _calc_prob(self, x):
        x = np.atleast_2d(x)
        rs = np.concatenate([self._as[k]*self._multi_gauss(k, x)
                for k in range(self._K)]).reshape((-1, self._K), order='F')
        L = np.log(np.sum(rs, axis=1))
        rs /= np.sum(rs, axis=1)[:,None]
        return L, rs

    def lnprob(self, x):
        return self._calc_prob(x)[0]

    def sample(self,N):
        samples = np.vstack(
                [np.random.multivariate_normal(self.means[k], self._cov[k],
                    size=int(self._as[k]*(N+1))) for k in range(self._K)])
        return samples[:N,:]

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as pl
    from matplotlib.patches import Ellipse

    np.random.seed(150)
    means     = np.array([[2.1,4.5],
                          [2.0, 2.7],
                          [3.5,5.6]])
    covariances = [np.array([[0.2,0.1], [0.1,0.6]]),
                   np.array([[0.35,0.22], [0.22,0.15]]),
                   np.array([[0.06,0.05], [0.05,1.3]])]
    amplitudes= [5,1,2]
    factor    = 100
    data = np.zeros((1,2))
    for i in range(len(means)):
        data = np.concatenate([data,
            np.random.multivariate_normal(means[i], covariances[i], size=factor*amplitudes[i])])
    data = data[1:,:]

    kmeans = MixtureModel(3, data)
    kmeans.run_kmeans()
    kmeans.run_em()

    samples = kmeans.sample(50000)

    pl.plot(samples[:,0], samples[:,1], '.k', zorder=-1, ms=2)
    pl.scatter(data[:,0], data[:,1], marker='o',
            c=[tuple(kmeans._rs[i,:]) for i in range(data.shape[0])],
            s=8., edgecolor='none')

    for k in range(kmeans.K):
        x,y = kmeans.means[k][0],kmeans.means[k][1]
        U,S,V = np.linalg.svd(kmeans._cov[k])
        theta = np.degrees(np.arctan2(U[1,0], U[0,0]))
        ellipsePlot = Ellipse(xy=[x,y], width=2*np.sqrt(S[0]),
            height=2*np.sqrt(S[1]), angle=theta,
            facecolor='none', edgecolor='w',lw=2)
        ax = pl.gca()
        ax.add_patch(ellipsePlot)


    pl.savefig('test.png')


