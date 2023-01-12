#!/usr/bin/env python3

import numpy as np
from scipy import stats
from ffpack.utils import gradient, hessianMatrix, gramSchmidOrth
from ffpack.rrm import formHLRF
from ffpack import rpm


def sormBreitung( dim, g, dg, distObjs, corrMat, iter=1000, tol=1e-6, 
                  quadDeg=99, quadRange=8, dx=1e-6 ):
    '''
    First order reliability method based on Breitung algorithm.

    Parameters
    ----------
    dim: integer
        Space dimension ( number of random variables ).
    g: function
        Limit state function. It will be called like g( [ x1, x2, ... ] ).
    dg: array_like of function 
        Gradient of the limit state function. It should be an array_like of function
        like dg = [ dg_dx1, dg_dx2, ... ]. To get the derivative of i-th random 
        variable at ( x1*, x2*, ... ), dg[ i ]( x1*, x2*, ... ) will be called.
        dg can be None, see the following Notes.
    distObjs: array_like of distributions
        Marginal distribution objects. It should be the freezed distribution 
        objects with pdf, cdf, ppf. We recommend to use scipy.stats functions.
    corrMat: 2d matrix
        Correlation matrix of the marginal distributions.
    iter: integer
        Maximum iteration steps.
    tol: scalar
        Tolerance to demtermine if the iteration converges.
    quadDeg: integer
        Quadrature degree for Nataf transformation
    quadRange: scalar
        Quadrature range for Nataf transformation. The integral will be performed 
        in the range [ -quadRange, quadRange ].
    dx : scalar, optional
        Spacing for auto differentiation. Not required if dg is provided.
    
    Returns
    -------
    beta: scalar
        Reliability index.
    pf: scalar
        Probability of failure.
    uCoord: 1d array
        Design point coordinate in U space.
    xCoord: 1d array
        Design point coordinate in X space.
    
    Raises
    ------
    ValueError
        If the dim is less than 1.
        If the dim does not match the disObjs and corrMat.
        If corrMat is not 2d matrix.
        If corrMat is not positive definite.
        If corrMat is not symmetric.
        If corrMat diagonal is not 1.

    Notes
    -----
    If dg is None, the numerical differentiation will be used. The tolerance of the 
    numerical differentiation can be changed in globalConfig.

    Examples
    --------
    >>> from ffpack.rrm import formHLRF
    >>> dim = 2
    >>> g = lambda X: -np.sum( X ) + 1
    >>> dg = [ lambda X: -1, lambda X: -1 ]
    >>> distObjs = [ stats.norm(), stats.norm() ]
    >>> corrMat = np.eye( dim )
    >>> beta, pf, uCoord, xCoord = formHLRF( dim, g, dg, distObjs, corrMat )
    '''
    beta, pf, uCoord, xCoord = formHLRF( dim, g, dg, distObjs, corrMat )

    if dg is None:
        dg = gradient( g, dim, n=1, dx=dx )
    
    lsfGradAtX = [ dgi( xCoord ) for dgi in dg ]
    
    hm = hessianMatrix( g, dim, dx=dx )
    lsfHmAtX = [ [ hmij( xCoord ) for hmij in hmi ] for hmi in hm ]
