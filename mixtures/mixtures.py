#!/usr/bin/env python
"""
Gaussian mixture models

"""

from __future__ import division

__all__ = ['MixtureModel']

import numpy as np
import numpy.ma as ma

import _algorithms

class MixtureModel(object):
    """
    Gaussian mixture model

    P data points in D dimensions with K clusters

    Shapes
    ------
    data -> (P, D)
    means -> (D, K)

    """
    def __init__(self, K, data, init_grid=False):
        self._K    = K
        self._data = np.atleast_2d(data)
        self._lu   = None

        if init_grid:
            inds = np.array(np.linspace(0, data.shape[0]-1, self._K), dtype=int)
        else:
            inds = np.random.randint(data.shape[0],size=self._K)
        self._means = data[inds,:]
        self._cov   = np.array([np.cov(data,rowvar=0)]*self._K)
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
        return self._means

    # ================= #
    # K-Means Algorithm #
    # ================= #

    @property
    def memberships(self):
        return self._kmeans_rs

    def run_kmeans(self, maxiter=200, tol=1e-4, verbose=True):
        self._kmeans_rs = np.zeros(self._data.shape[0], dtype=int)
        _algorithms.kmeans(self._data, self._means, self._kmeans_rs, tol, maxiter)

    def get_hist(self):
        h = np.array([np.sum(self._kmeans_rs == k) for k in range(self._K)])
        return h/np.sum(h)

    def get_entropy(self):
        h = self.get_hist()
        inds = h > 0
        return -np.sum(h[inds]*np.log2(h[inds]))

    def get_max_entropy(self):
        return -np.log2(1.0/self._K)

    # ============ #
    # EM Algorithm #
    # ============ #

    def run_em(self, maxiter=400, tol=1e-4, verbose=True, normalize=None):
        if normalize is not None:
            self._means /= normalize
            self._data /= normalize

        try:
            _algorithms.em(self._data, self._means, self._cov, self._as, tol, maxiter)
        except AttributeError: # not compiled with LAPACK
            self.run_em_slow(maxiter=maxiter, tol=tol, verbose=verbose, normalize=normalize)

    def run_em_slow(self, maxiter=400, tol=1e-4, verbose=True, normalize=None):
        """
        Fit the given data using EM

        """
        self._means = self._means.T

        L = None
        for i in xrange(maxiter):
            newL = self._expectation()
            if i == 0:
                print "Initial NLL =", -newL
            self._maximization()
            if L is None:
                L = newL
            else:
                dL = np.abs((newL-L)/L)
                if i > 5 and dL < tol:
                    break
                L = newL
        if i < maxiter-1:
            if verbose:
                print "EM converged after %d iterations"%(i)
                print "Final NLL =", -newL
        else:
            print "Warning: EM didn't converge after %d iterations"%(i)

        self._means = self._means.T

    def _log_multi_gauss(self, k, X):
        # X.shape == (P,D)
        # self._means.shape == (D,K)
        # self.cov[k].shape == (D,D)
        sgn, logdet = np.linalg.slogdet(self._cov[k])
        if sgn <= 0:
            return -np.inf*np.ones(X.shape[0])

        # X1.shape == (P,D)
        X1 = X - self._means[None,:,k]

        # X2.shape == (P,D)
        X2 = np.linalg.solve(self._cov[k], X1.T).T

        p = -0.5*np.sum(X1 * X2, axis=1)

        return -0.5 * np.log( (2*np.pi)**(X.shape[1]) ) - 0.5 * logdet + p

    def _expectation(self):
        # self._rs.shape == (P,K)
        L, self._rs = self._calc_prob(self._data)
        return np.sum(L, axis=0)

    def _maximization(self):
        # Nk.shape == (K,)
        Nk = np.sum(self._rs, axis=0)
        Nk = ma.masked_array(Nk, mask=Nk<=0)
        # self._means.shape == (D,K)
        self._means = ma.masked_array(np.sum(self._rs[:,None,:] \
                * self._data[:,:,None], axis=0))
        self._means /= Nk[None, :]
        self._cov = []
        for k in range(self._K):
            # D.shape == (P,D)
            D = self._data - self._means[None,:,k]
            self._cov.append(np.dot(D.T, self._rs[:,k,None]*D)/Nk[k])
        self._as = Nk/self._data.shape[0]

    def _calc_prob(self, x):
        x = np.atleast_2d(x)

        logrs = []
        for k in range(self._K):
            logrs += [np.log(self._as[k]) + self._log_multi_gauss(k, x)]
        logrs = np.concatenate(logrs).reshape((-1, self._K), order='F')

        # here lies some ghetto log-sum-exp...
        # nothing like a little bit of overflow to make your day better!
        a = np.max(logrs, axis=1)
        L = a + np.log(np.sum(np.exp(logrs-a[:,None]), axis=1))
        logrs -= L[:,None]
        return L, np.exp(logrs)

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

    mixture = MixtureModel(3, data)
    mixture.run_kmeans()
    mixture.run_em()

    pl.scatter(data[:,0], data[:,1], marker='o',
            c=[tuple(mixture._rs[i,:]) for i in range(data.shape[0])],
            s=8., edgecolor='none')

    for k in range(mixture.K):
        x,y = mixture.means[k][0],mixture.means[k][1]
        U,S,V = np.linalg.svd(mixture._cov[k])
        theta = np.degrees(np.arctan2(U[1,0], U[0,0]))
        ellipsePlot = Ellipse(xy=[x,y], width=2*np.sqrt(S[0]),
            height=2*np.sqrt(S[1]), angle=theta,
            facecolor='none', edgecolor='k',lw=2)
        ax = pl.gca()
        ax.add_patch(ellipsePlot)


    pl.savefig('em-test.png')


