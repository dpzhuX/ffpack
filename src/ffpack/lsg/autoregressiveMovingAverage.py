#!/usr/bin/env python3

import numpy as np


def arNormal( numSteps, obs, phis, mu, sigma, randomSeed=None ):
    '''
    Generate load sequence by an autoregressive model.

    The white noise is generated by the normal distribution.

    Parameters
    ----------
    numSteps: integer 
        Number of steps for generating.
    obs: 1d array
        Initial observed values.
    phis: 1d array
        Coefficients for the autoregressive model.
    mu: scalar
        Mean of the white noise.
    sigma: scalar
        Standard deviation of the white noise.
    randomSeed: integer, optional
        Random seed. If randomSeed is none or is not an integer, the random seed in 
        global config will be used. 
    
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
    >>> obs = [ 0, 1 ]
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
    if isinstance(randomSeed, (int, type(None))):
        np.random.seed( randomSeed )
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


def maNormal( numSteps, c, thetas, mu, sigma, randomSeed=None ):
    '''
    Generate load sequence by a moving-average model.

    The white noise is generated by the normal distribution.

    Parameters
    ----------
    numSteps: integer 
        Number of steps for generating.
    c: scalar
        Mean of the series.
    thetas: 1d array
        Coefficients for the white noise in the moving-average model.
    mu: scalar
        Mean of the white noise.
    sigma: scalar
        Standard deviation of the white noise.
    randomSeed: integer, optional
        Random seed. If randomSeed is none or is not an integer, the random seed in 
        global config will be used. 
    
    Returns
    -------
    rst: 1d array
        Generated sequence with moving-average model.
    
    Raises
    ------
    ValueError
        If the numSteps is less than 1.
        If mean of the series is not a scalar.
        If the thetas is empty.

    Examples
    --------
    >>> from ffpack.lsg import maNormal
    >>> thetas = [ 0.8, 0.5 ]
    >>> rst = maNormal( 500, 0, thetas, 0, 0.5 )
    '''
    # Edge case check
    if not isinstance( numSteps, int ):
        raise ValueError( "numSteps should be int" )
    if numSteps < 1:
        raise ValueError( "numSteps should be at least 1" )
    if not isinstance( c, int ) and not isinstance( c, float ):
        raise ValueError( "mean of the series should be a scalar" )
    if len( thetas ) < 1:
        raise ValueError( "length of coefficients for the white noise should be at least 1" )

    if isinstance(randomSeed, (int, type(None))):
        np.random.seed( randomSeed )
    eps = np.random.normal( mu, sigma, numSteps )

    rst = [ 0 ] * numSteps
    q = len( thetas )
    for i in range( numSteps ):
        ept = eps[ i ]
        for j in range( q ):
            if ( i > j ): 
                ept += thetas[ j ] * eps[ i - j - 1 ]
        rst[ i ] = c + ept
    
    return rst


def armaNormal( numSteps, obs, phis, thetas, mu, sigma, randomSeed=None ):
    '''
    Generate load sequence by an autoregressive-moving-average model.

    The white noise is generated by the normal distribution.

    Parameters
    ----------
    numSteps: integer 
        Number of steps for generating.
    obs: 1d array
        Initial observed values, could be empty.
    phis: 1d array
        Coefficients for the autoregressive part.
    thetas: 1d array
        Coefficients for the white noise for the moving-average part.
    mu: scalar
        Mean of the white noise.
    sigma: scalar
        Standard deviation of the white noise.
    randomSeed: integer, optional
        Random seed. If randomSeed is none or is not an integer, the random seed in 
        global config will be used. 
    
    Returns
    -------
    rst: 1d array
        Generated sequence includes the observed values.
    
    Raises
    ------
    ValueError
        If the numSteps is less than 1.
        If the phis is empty.
        If the thetas is empty.

    Examples
    --------
    >>> from ffpack.lsg import armaNormal
    >>> obs = [ 0, 1 ]
    >>> phis = [ 0.5, 0.3 ]
    >>> thetas = [ 0.8, 0.5 ]
    >>> rst = armaNormal( 500, obs, phis, thetas, 0, 0.5 )
    '''
    # Edge case check
    if not isinstance( numSteps, int ):
        raise ValueError( "numSteps should be int" )
    if numSteps < 1:
        raise ValueError( "numSteps should be at least 1" )
    if len( phis ) < 1:
        raise ValueError( "length of phis should be at least 1" )
    if len( thetas ) < 1:
        raise ValueError( "length of coefficients for the white noise should be at least 1" )

    p = len( phis )
    q = len( thetas )
    n = len( obs )
    if isinstance(randomSeed, (int, type(None))):
        np.random.seed( randomSeed )
    eps = np.random.normal( mu, sigma, numSteps )

    rst = [ 0 ] * numSteps
    if ( n > 0 ):
        for i in range( min( n, numSteps ) ):
            rst[ i ] = obs[ i ]
    
    for i in range( n, numSteps ):
        epa = 0
        for j in range( p ):
            if ( i > j ): 
                epa += phis[ j ] * rst[ i - j - 1 ]
        
        epm = 0
        for j in range( q ):
            if ( i > j ): 
                epm += thetas[ j ] * eps[ i - j - 1 ]
        
        rst[ i ] = eps[ i ] + epa + epm 

    return rst


def arimaNormal( numSteps, c, phis, thetas, mu, sigma, randomSeed=None ):
    '''
    Generate load sequence by an autoregressive integrated moving average model.

    The white noise is generated by the normal distribution.

    First-order diference is used in this function.

    Parameters
    ----------
    numSteps: integer 
        Number of steps for generating.
    c: scalar
        Mean of the series.
    phis: 1d array
        Coefficients for the autoregressive part.
    thetas: 1d array
        Coefficients for the white noise for the moving-average part.
    mu: scalar
        Mean of the white noise.
    sigma: scalar
        Standard deviation of the white noise.
    randomSeed: integer, optional
        Random seed. If randomSeed is none or is not an integer, the random seed in 
        global config will be used. 
        
    Returns
    -------
    rst: 1d array
        Generated sequence with the autoregressive integrated moving average model.
    
    Raises
    ------
    ValueError
        If the numSteps is less than 1.
        If mean of the series is not a scalar.
        If the phis is empty.
        If the thetas is empty.

    Examples
    --------
    >>> from ffpack.lsg import arimaNormal
    >>> phis = [ 0.5, 0.3 ]
    >>> thetas = [ 0.8, 0.5 ]
    >>> rst = arimaNormal( 500, 0.0, phis, thetas, 0, 0.5 )
    '''
    # Edge case check
    if not isinstance( numSteps, int ):
        raise ValueError( "numSteps should be int" )
    if numSteps < 1:
        raise ValueError( "numSteps should be at least 1" )
    if not isinstance( c, int ) and not isinstance( c, float ):
        raise ValueError( "mean of the series should be a scalar" )
    if len( phis ) < 1:
        raise ValueError( "length of phis should be at least 1" )
    if len( thetas ) < 1:
        raise ValueError( "length of coefficients for the white noise should be at least 1" )

    p = len( phis )
    q = len( thetas )
    if isinstance(randomSeed, (int, type(None))):
        np.random.seed( randomSeed )
    eps = np.random.normal( mu, sigma, numSteps )

    rst = [ 0 ] * numSteps
    
    for i in range( numSteps ):
        epa = 0
        for j in range( p ):
            if ( i > j + 1 ): 
                epa += phis[ j ] * ( rst[ i - j - 1 ] - rst[ i - j - 2 ] )
        
        epm = 0
        for j in range( q ):
            if ( i > j ): 
                epm += thetas[ j ] * eps[ i - j - 1 ]
        
        rst[ i ] = c + eps[ i ] + epa + epm 

    return rst
