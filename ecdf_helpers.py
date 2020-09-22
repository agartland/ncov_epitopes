import pandas as pd
import numpy as np
import matplotlib as mpl

__all__ = ['compute_ecdf',
           'make_step',
           'gmean10',
           'color_lu',
           'norm_pgen']

def compute_ecdf(data, counts=None, thresholds=None):
    """Computes the empirical cumulative distribution function at pre-specified
    thresholds. Assumes thresholds is sorted and should be unique."""
    if thresholds is None:
        thresholds = np.unique(data[:])
    if counts is None:
        counts = np.ones(data.shape)
    
    tot = np.sum(counts)
    # ecdf = np.array([np.sum((data <= t) * counts)/tot for t in thresholds])
    
    """Vectorized and faster, using broadcasting for the <= expression"""
    ecdf = (np.sum((data[:, None] <= thresholds[None, :]) * counts[:, None], axis=0) + 1) / (tot + 1)
    # n_ecdf = (np.sum((data[:, None] <= thresholds[None, :]) * counts[:, None], axis=0) >= n).astype(int)
    return ecdf

def make_step(t, y, add00=False, addMNMN=False):
    y = np.asarray(y)
    t = np.asarray(t)
    if add00:
        t = np.concatenate(([0], t.ravel()))
        y = np.concatenate(([0], y.ravel()))
    elif addMNMN:
        pass
        #t = np.concatenate(([0], t.ravel()))
        #y = np.concatenate(([0], y.ravel()))

    t = np.concatenate(([t[0]], np.repeat(t[1:].ravel(), 2)))
    y = np.repeat(y.ravel(), 2)[:-1]
    return t, y

def gmean10(vec, axis=0):
    return 10 ** (np.mean(np.log10(vec), axis=axis))


norm_pgen = mpl.colors.LogNorm(vmin=1e-10, vmax=1e-6) 

def color_lu(norm, colors, pgen):
    i = int(np.floor(norm(pgen) * len(colors)))
    if i >= len(colors):
        i = len(colors) - 1
    if i < 0:
        i = 0
    return tuple(colors[i])
