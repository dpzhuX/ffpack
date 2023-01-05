#!/usr/bin/env python3

import numpy as np
import scipy as sp
from ffpack.config import globalConfig 
from ffpack import rpm

def form( dim, g, dg, distObjs, corrMat, iter=1000, tol=1e-6 ):
    '''
    Relibility based on first order reliability method.

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
    >>> from ffpack.rrm import form
    >>> dim = 2
    >>> g = lambda X: 3 * X[ 0 ] - 2 * X[ 1 ]
    >>> dg = [ lambda X: 3, lambda X: -2 ]
    >>> mus = [ 1, 1 ]
    >>> sigmas = [ 3, 4 ]
    >>> beta, pf = fosm( dim, g, dg, mus, sigmas)
    '''
    if dim < 1:
        raise ValueError( "dim cannot be less than 1" )

    corrMat = np.array( corrMat )
    if not np.all( np.diag( corrMat ) == 1 ):
        raise ValueError( "diagonals of corrMat should be 1" )

    if len( distObjs ) != dim or corrMat.shape[ 0 ] != dim \
            or corrMat.shape[ 1 ] != dim: 
        raise ValueError( "length of distObjs and corrMat should be dim" )

    if corrMat.ndim != 2:
        raise ValueError( "corrMat should be 2d matrix" )

    if not np.array_equal( corrMat, corrMat.T ):
        raise ValueError( "corrMat should be symmetric" )
    
    try:
        _ = np.linalg.cholesky( corrMat )
    except np.linalg.LinAlgError:
        raise ValueError( "corrMat should be positive definite" )

    def partialDerivative( func, var=0, points=[ ] ):
        args = points[ : ]

        def wraps( x ):
            args[ var ] = x
            return func( args )
        return sp.misc.derivative( wraps, points[ var ], 
                                   dx=1 / np.power( 10, globalConfig.dtol ) )
    
    def dgWrap( g, var=0 ):
        def dgi( mus ):
            return partialDerivative( g, var=var, points=mus )
        return dgi

    if dg is None:
        dg = [ dgWrap( g, i ) for i in range( dim ) ]

    natafTrans = rpm.NatafTransformation( distObjs=distObjs, corrMat=corrMat )

    Us = np.ones( [ iter + 1, dim ] )
    alphas = np.zeros_like( Us )
    betas = np.zeros( iter + 1 )
    idx = 1
    for idx in range( 1, iter + 1):
        # J: U -> X is partialX / partialU
        X, J = natafTrans.getX( Us[ idx - 1 ] )

        a = np.array( [ dgi( X ) for dgi in dg ] )
        gPrime = np.linalg.solve( J.T, a )
        gPrime = gPrime.T.flatten()
        gPrimeNorm = np.linalg.norm( gPrime )

        betas[ idx ] = ( g( X ) - np.dot( Us[ idx - 1 ], gPrime ) ) / gPrimeNorm

        alphas[ idx ] = gPrime / gPrimeNorm

        Us[ idx ] = -betas[ idx ] * alphas[ idx ]
        if np.linalg.norm( Us[ idx ] - Us[ idx - 1 ] ) < tol:
            break
    
    beta = betas[ idx ]
    pf = sp.stats.norm.cdf( -beta )
    uCoord = Us[ idx ]
    xCoord, _ = natafTrans.getX( uCoord )
    return beta, pf, uCoord, xCoord
