#!/usr/bin/env python3

import numpy as np
from scipy import stats
from ffpack.utils import gradient

def fosmMVAL( dim, g, dg, mus, sigmas, dx=1e-6 ):
    '''
    First order second moment method based on mean value algorithm.

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
    mus: 1d array
        Mean of the random variables.
    sigmas: 1d array
        Variance of the random variables.
    dx : scalar, optional
        Spacing for auto differentiation. Not required if dg is provided.
    
    Returns
    -------
    beta: scalar
        Reliability index.
    pf: scalar
        probability of failure.
    
    Raises
    ------
    ValueError
        If the dim is less than 1.
        If the dim does not match the length of mus and sigmas.

    Notes
    -----
    If dg is None, the numerical differentiation will be used. The tolerance of the 
    numerical differentiation can be changed in globalConfig.

    Examples
    --------
    >>> from ffpack.rrm import fosmMVAL
    >>> dim = 2
    >>> g = lambda X: 3 * X[ 0 ] - 2 * X[ 1 ]
    >>> dg = [ lambda X: 3, lambda X: -2 ]
    >>> mus = [ 1, 1 ]
    >>> sigmas = [ 3, 4 ]
    >>> beta, pf = fosmMVAL( dim, g, dg, mus, sigmas)
    '''
    if dim < 1:
        raise ValueError( "dim cannot be less than 1" )
    if len( mus ) != dim or len( sigmas ) != dim:
        raise ValueError( "length of mus and sigmas should be dim" )

    if dg is None:
        dg = gradient( g, dim, n=1, dx=dx )

    lsfRst = g( mus )
    a = np.array( [ dgi( mus ) for dgi in dg ] )
    aSigmas = np.multiply( a, sigmas )
    beta = lsfRst / np.sqrt( np.sum( np.square( aSigmas ) ) )
    pf = stats.norm.cdf( -beta )

    return beta, pf
