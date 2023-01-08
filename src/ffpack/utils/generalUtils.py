#!/usr/bin/env python3

import numpy as np

def sequencePeakAndValleys( data, keepEnds=False ):
    '''
    Remove the intermediate value and only get the peaks and valleys of the data

    The peak and valley refer the data points that are EXACTLY above and below
    the neighbors, not equal. 

    Parameters
    ----------
    data: 1darray
        Sequence data to get peaks and valleys.
    
    keepEnds: bool, optional
        If two ends of the original data should be preserved.
    
    Returns
    -------
    rst: 1darray
        A list contains the peaks and valleys of the data.
    
    Raises
    ------
    ValueError
        If the data dimension is not 1.
        If the data length is less than 2 with keedEnds == False
        If the data length is less than 3 with keedEnds == True

    Examples
    --------
    >>> from ffpack.utils import sequencePeakAndValleys
    >>> data = [ -0.5, 1.0, -2.0, 3.0, -1.0, 4.5, -2.5, 3.5, -1.5, 1.0 ]
    >>> rst = sequencePeakAndValleys( data )
    '''
    # Egde cases
    data = np.array( data )
    if len( data.shape ) != 1:
        raise ValueError( "Input data dimension should be 1" )
    if data.shape[0] <= 1 and keepEnds:
        raise ValueError( "Input data length should be at least 2")
    if data.shape[0] <= 2 and not keepEnds:
        raise ValueError( "Input data length should be at least 3")

    rst = []
    prev = data[ 0 ]
    for i, cur in enumerate( data ):
        if i == 0 or i == len( data ) - 1:
            if keepEnds:
                rst.append( cur )
        else:
            next = data[ i + 1 ]
            if ( prev < cur and cur > next ) or \
               ( prev > cur and cur < next ):
                rst.append( cur )
                prev = cur
        
    return rst

def sequenceDigitization( data, resolution=1.0 ):
    '''
    Digitize the sequence data to a specific resolution

    The sequence data are digitized by the round method. 

    Parameters
    ----------
    data: 1d array
        Sequence data to digitize.
    
    resolution: bool, optional
        The desired resolution to round the data points.
    
    Returns
    -------
    rst: 1d array
        A list contains the digitized data.
    
    Raises
    ------
    ValueError
        If the data dimension is not 1.
        If the data length is less than 2 with keedEnds == False
        If the data length is less than 3 with keedEnds == True

    Notes
    -----
    The default round function will round half to even: 1.5, 2.5 => 2.0:

    Examples
    --------
    >>> from ffpack.utils import sequenceDigitization 
    >>> data = [ -1.0, 2.3, 1.8, 0.6, -0.4, 0.8, -1.6, -2.5, 3.4, 0.3, 0.1 ]
    >>> rst = sequenceDigitization( data )
    '''
    # Egde cases
    data = np.array( data )
    if len( data.shape ) != 1:
        raise ValueError( "Input data dimension should be 1" )

    rst = []
    for d in data:
        rst.append( np.rint( d / resolution) * resolution )
    return rst


def centralDiffWeights( Np, ndiv=1 ):
    """
    Return weights for an Np-point central derivative [1]_.

    Assumes equally-spaced function points.

    If weights are in the vector w, then
    derivative is w[ 0 ] * f( x - ho * dx ) + ... + w[ -1 ] * f( x + h0 * dx )

    Parameters
    ----------
    Np : integer
        Number of points for the central derivative.
    ndiv : integer, optional
        Number of divisions. Default is 1.

    Returns
    -------
    w : ndarray
        Weights for an Np-point central derivative. Its size is `Np`.

    Notes
    -----
    Can be inaccurate for a large number of points.
    This function came from scipy.misc module, we put it here since scipy.misc module 
    is completely removed in SciPy v1.12.0.

    Examples
    --------
    >>> def f( x ):
    ...     return 2 * x**2 + 3
    >>> x = 3.0 # derivative point
    >>> h = 0.1 # differential step
    >>> Np = 3 # point number for central derivative
    >>> weights = centralDiffWeights( Np ) # weights for first derivative
    >>> vals = [ f( x + ( i - Np / 2 ) * h) for i in range( Np )]
    >>> sum( w * v for (w, v) in zip( weights, vals ) ) / h
    11.79999999999998
    This value is close to the analytical solution:
    f'(x) = 4x, so f'(3) = 12

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Finite_difference
    """
    if Np < ndiv + 1:
        raise ValueError(
            "Number of points must be at least the derivative order + 1."
        )
    if Np % 2 == 0:
        raise ValueError( "The number of points must be odd." )
    from scipy import linalg

    ho = Np >> 1
    x = np.arange(-ho, ho + 1.0)
    x = x[ :, np.newaxis ]
    X = x**0.0
    for k in range( 1, Np ):
        X = np.hstack( [ X, x**k ] )
    w = np.prod( np.arange( 1, ndiv + 1 ), axis=0 ) * linalg.inv( X )[ ndiv ]
    return w


def derivative( func, x0, dx=1.0, n=1, args=(), order=3 ):
    """
    Find the nth derivative of a function at a point.

    Given a function, use a central difference formula with spacing `dx` to
    compute the nth derivative at `x0`.

    Parameters
    ----------
    func : function
        Input function.
    x0 : float
        The point at which the nth derivative is found.
    dx : float, optional
        Spacing.
    n : int, optional
        Order of the derivative. Default is 1.
    args : tuple, optional
        Arguments
    order : int, optional
        Number of points to use, must be odd.

    Notes
    -----
    Decreasing the step size too small can result in round-off error.
    This function came from scipy.misc module, we put it here since scipy.misc module 
    is completely removed in SciPy v1.12.0.

    Examples
    --------
    >>> def f(x):
    ...     return x**3 + x**2
    >>> _derivative( f, 1.0, dx=1e-6 )
    4.9999999999217337
    """

    if order < n + 1:
        raise ValueError(
            "'order' (the number of points used to compute the derivative), "
            "must be at least the derivative order 'n' + 1."
        )

    if order % 2 == 0:
        raise ValueError(
            "'order' (the number of points used to compute the derivative) "
            "must be odd."
        )
    # pre-computed for n=1 and 2 and low-order for speed.
    if n == 1:
        if order == 3:
            weights = np.array( [ -1, 0, 1 ] ) / 2.0
        elif order == 5:
            weights = np.array( [ 1, -8, 0, 8, -1 ] ) / 12.0
        elif order == 7:
            weights = np.array( [ -1, 9, -45, 0, 45, -9, 1 ] ) / 60.0
        elif order == 9:
            weights = np.array( [ 3, -32, 168, -672, 0, 672, -168, 32, -3 ] ) / 840.0
        else:
            weights = centralDiffWeights( order, 1 )
    elif n == 2:
        if order == 3:
            weights = np.array( [ 1, -2.0, 1 ] )
        elif order == 5:
            weights = np.array( [ -1, 16, -30, 16, -1 ] ) / 12.0
        elif order == 7:
            weights = np.array( [ 2, -27, 270, -490, 270, -27, 2 ] ) / 180.0
        elif order == 9:
            weights = (
                np.array( [ -9, 128, -1008, 8064, -14350, 
                            8064, -1008, 128, -9 ] ) / 5040.0
            )
        else:
            weights = centralDiffWeights( order, 2 )
    else:
        weights = centralDiffWeights( order, n )
    val = 0.0
    ho = order >> 1
    for k in range( order ):
        val += weights[ k ] * func( x0 + ( k - ho ) * dx, *args )
    return val / np.prod( ( dx, ) * n, axis=0 )
