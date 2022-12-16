#!/usr/bin/env python3

import numpy as np

def arNormal( numSteps, obs, phis, mu, sigma ):
    '''
    Generate load sequence by autoregressive model.

    The white noise is generated by the normal distribution.

    Parameters
    ----------
    numSteps: integer 
        Number of steps for generating.
    obs: 1d array
        Initial observed values.
    phis: 1d array
        Coefficients for autoregressive model.
    mu: scalar
        Mean of the white noise.
    sigma: scalar
        Standard deviation of the white noise.
    
    Returns
    -------
    rst: 1d array
        Generated sequence includes the observed values.
    
    Raises
    ------
    ValueError
        If the numSteps is less than 1.
        If lengths of obs and phis are not equal.

    Examples
    --------
    >>> from ffpack.lsg import arNormal
    >>> obs = [ 0, 0  ]
    >>> phis = [ 0.5, 0.3 ]
    >>> rst = arNormal( 500, obs, phis, 0, 0.5 )
    '''
    # Edge case check
    if not isinstance( numSteps, int ):
        raise ValueError( "numSteps should be int" )
    if numSteps < 1:
        raise ValueError( "numSteps should be at least 1" )
    if len( obs ) != len( phis ):
        raise ValueError( "lengths of obs and phis should be same" )
    if len( obs ) < 1:
        raise ValueError( "length of obs or phis should be at least 1" )

    p = len( obs )
    eps = np.random.normal( mu, sigma, numSteps )

    rst = [ 0 ] * numSteps
    for i in range( numSteps ):
        if i < p:
            rst[ i ] = obs[ i ]
        else:
            rst[ i ] += eps[ i ]
            for j in range( p ):
                rst[ i ] += phis[ j ] * rst[ i - j - 1]
    
    return rst
