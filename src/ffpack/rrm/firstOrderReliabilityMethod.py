#!/usr/bin/env python3

import numpy as np
from scipy import stats, optimize
from ffpack.utils import gradient
from ffpack import rpm

def hlrfFORM( dim, g, dg, distObjs, corrMat, iter=1000, tol=1e-6, 
              quadDeg=99, quadRange=8, dx=1e-6 ):
    '''
    First order reliability method based on Hasofer-Lind-Rackwitz-Fiessler algorithm.

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
    >>> from ffpack.rrm import hlrfFORM
    >>> dim = 2
    >>> g = lambda X: -np.sum( X ) + 1
    >>> dg = [ lambda X: -1, lambda X: -1 ]
    >>> distObjs = [ stats.norm(), stats.norm() ]
    >>> corrMat = np.eye( dim )
    >>> beta, pf, uCoord, xCoord = hlrfFORM( dim, g, dg, distObjs, corrMat )
    '''
    # Check edge cases
    if dim < 1:
        raise ValueError( "dim cannot be less than 1" )

    corrMat = np.array( corrMat, dtype=float )
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

    if dg is None:
        dg = gradient( g, dim, n=1, dx=dx )

    natafTrans = rpm.NatafTransformation( distObjs=distObjs, corrMat=corrMat, 
                                          quadDeg=quadDeg, quadRange=quadRange )

    Us = np.ones( [ iter + 1, dim ] )
    alphas = np.zeros_like( Us )
    betas = np.zeros( iter + 1 )
    idx = 1
    for idx in range( 1, iter + 1):
        # J: U -> X is partialX / partialU
        X, J = natafTrans.getX( Us[ idx - 1 ] )

        a = np.array( [ dgi( X ) for dgi in dg ], dtype=float )
        gPrime = np.linalg.solve( J.T, a.reshape( -1, 1) )
        gPrime = gPrime.T.flatten()
        gPrimeNorm = np.linalg.norm( gPrime )

        betas[ idx ] = ( g( X ) - np.dot( Us[ idx - 1 ], gPrime ) ) / gPrimeNorm

        alphas[ idx ] = gPrime / gPrimeNorm

        Us[ idx ] = -betas[ idx ] * alphas[ idx ]
        if np.linalg.norm( Us[ idx ] - Us[ idx - 1 ] ) < tol:
            break
    
    # We do not expect convergence with iter == 1
    if iter != 1 and np.linalg.norm( Us[ idx ] - Us[ idx - 1 ] ) >= tol:
        raise ValueError( "hlrfFORM does not converge with current parameters.")

    beta = betas[ idx ]
    pf = stats.norm.cdf( -beta )
    uCoord = Us[ idx ]
    xCoord, _ = natafTrans.getX( uCoord )
    return beta, pf, uCoord.tolist(), xCoord.tolist()


def coptFORM( dim, g, distObjs, corrMat, quadDeg=99, quadRange=8 ):
    '''
    First order reliability method based on constrained optimization.

    Parameters
    ----------
    dim: integer
        Space dimension ( number of random variables ).
    g: function
        Limit state function. It will be called like g( [ x1, x2, ... ] ).
    distObjs: array_like of distributions
        Marginal distribution objects. It should be the freezed distribution 
        objects with pdf, cdf, ppf. We recommend to use scipy.stats functions.
    corrMat: 2d matrix
        Correlation matrix of the marginal distributions.
    quadDeg: integer
        Quadrature degree for Nataf transformation
    quadRange: scalar
        Quadrature range for Nataf transformation. The integral will be performed 
        in the range [ -quadRange, quadRange ].
    
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

    Examples
    --------
    >>> from ffpack.rrm import coptFORM
    >>> dim = 2
    >>> g = lambda X: -np.sum( X ) + 1
    >>> distObjs = [ stats.norm(), stats.norm() ]
    >>> corrMat = np.eye( dim )
    >>> beta, pf, uCoord, xCoord = coptFORM( dim, g, distObjs, corrMat )
    '''
    # Check edge cases
    if dim < 1:
        raise ValueError( "dim cannot be less than 1" )

    corrMat = np.array( corrMat, dtype=float )
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
    natafTrans = rpm.NatafTransformation( distObjs=distObjs, corrMat=corrMat,
                                          quadDeg=quadDeg, quadRange=quadRange )

    u = np.ones( dim )

    def f( U ):
        return np.linalg.norm( U )

    cons = ( { "type": "eq",
               "fun": lambda U: g( natafTrans.getX( U )[ 0 ] ) } )
    
    rst = optimize.minimize( f, u, constraints=cons )
    beta = rst.fun
    pf = stats.norm.cdf( -beta )
    uCoord = rst.x
    xCoord = natafTrans.getX( uCoord )[ 0 ]
    return beta, pf, uCoord.tolist(), xCoord.tolist()
