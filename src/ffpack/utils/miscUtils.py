#!/usr/bin/env python3

import numpy as np


def centralDiffWeights( Np, ndiv=1 ):
    """
    Return weights for an Np-point central derivative [1]_.

    This function came from scipy.misc module, we put it here since scipy.misc module 
    is completely removed in SciPy v1.12.0.

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
    x = np.arange( -ho, ho + 1.0 )
    x = x[ :, np.newaxis ]
    X = x**0.0
    for k in range( 1, Np ):
        X = np.hstack( [ X, x ** k ] )
    w = np.prod( np.arange( 1, ndiv + 1 ), axis=0 ) * linalg.inv( X )[ ndiv ]
    return w


def derivative( func, x0, dx=1.0, n=1, args=(), order=3 ):
    """
    Find the n-th derivative of a function at a point.

    This function came from scipy.misc module, we put it here since scipy.misc module 
    is completely removed in SciPy v1.12.0.

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

    Examples
    --------
    >>> def f(x):
    ...     return x**3 + x**2
    >>> derivative( f, 1.0, dx=1e-6 )
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


def gradient( func, nvar, n=1, dx=1e-3, order=3 ):
    r'''
    Find n-th gradient of a scalar-valued differentiable function.

    Parameters
    ----------
    func : function
        Input scalar-valued differentiable function.
    nvar : integer
        The number of input variables for the input function. Input function will be 
        called like func( X ) = func( [ X[ 0 ], X[ 1 ], ..., X[ nvar - 1 ] ] ).
    n : integer, optional
        Order of the derivative. Default is 1.
    dx : scalar, optional
        Spacing for derivative calculation.
    order : integer, optional
        Number of points used for central derivative weights, must be odd.

    Returns
    -------
    rst : 1d array
        n-th gradient of function, i.e., [ :math:`\partial^n f / \partial X_0^n, 
        \dots, \partial^n f / \partial X_{nvar}^n` ]. In general, the i-th element 
        in the list is the n-th derivative of the func w.r.t. i-th input variable. 
        Therefore, rst[ i ] = :math:`\partial^n f / \partial X_i^n` and it can be
        called like rst[ i ]( X0 ) to evaluate the n-th derivative w.r.t. 
        i-th variable at point X0.

    Notes
    -----
    Decreasing the step size too small can result in round-off error.

    Examples
    --------
    >>> def f( X ):
    ...     return X[ 0 ]**3 + X[ 1 ]**2
    >>> gradient( f, nvar=2, n=1 )
    Output will be a function list of the 1st derivative of func
    >>> [ lambda X: 3 * X[ 0 ]**2, lambda X: 2 * X[ 1 ] ]
    >>> gradient( f, nvar=2, n=2 )
    Output will be a function list of the 2nd derivative of func
    >>> [ lambda X: 6 * X[ 0 ], lambda X: 2 ]
    '''
    if not isinstance( nvar, int ):
        raise ValueError( "nvar should be integer. " )

    if not isinstance( n, int ):
        raise ValueError( "n should be integer. " )

    if not isinstance( order, int ):
        raise ValueError( "order should be integer. " )

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

    # Evaluate n-th partial derivative w.r.t. vpos variable at points
    def partialDerivative( func, vpos=0, n=n, points=[ ] ):
        args = points[ : ]

        def wraps( x ):
            args[ vpos ] = x
            return func( args )
        return derivative( wraps, points[ vpos ], dx=dx, n=n, order=order )
    
    def dfWrap( func, vpos=0 ):
        def dfi( mus ):
            return partialDerivative( func, vpos=vpos, points=mus )
        return dfi

    rst = [ dfWrap( func, i ) for i in range( nvar ) ]
    return rst


def hessianMatrix( func, nvar, dx=1e-3, order=3 ):
    r'''
    Find Hessian matrix for a scalar-valued differentiable function.

    Parameters
    ----------
    func : function
        Input scalar-valued differentiable function.
    nvar : integer
        The number of input variables for the input function. Input function will be 
        called like func( X ) = func( [ X[ 0 ], X[ 1 ], ..., X[ nvar - 1 ] ] ).
    dx : scalar, optional
        Spacing for derivative calculation.
    order : integer, optional
        Number of points used for central derivative weights, must be odd.

    Returns
    -------
    rst : 2d array
        Hessian matrix. rst[ i ][ j ] = :math:`\partial f / 
        ( \partial X_i \partial X_j )`.  It can be called like rst[ i ][ j ]( X0 ) 
        to evaluate the value at point X0.

    Notes
    -----
    Decreasing the step size too small can result in round-off error.

    Examples
    --------
    >>> def f( X ):
    ...     return X[ 0 ]**3 + X[ 1 ]**2
    >>> hessianMatrix( f, nvar=2 )
    Output will be a function list of the 1st derivative of func
    >>> [ [ lambda X: 6 * X[ 0 ], lambda X: 0 ],
    ...   [ lambda X: 0, lambda X: 2] ]
    '''
    # Check edge cases
    if not isinstance( nvar, int ):
        raise ValueError( "nvar should be integer. " )

    if not isinstance( order, int ):
        raise ValueError( "order should be integer. " )

    if order < 2:
        raise ValueError(
            "'order' (the number of points used to compute the derivative), "
            "must be at least the derivative order of 2."
        )

    if order % 2 == 0:
        raise ValueError(
            "'order' (the number of points used to compute the derivative) "
            "must be odd."
        )
    grad = gradient( func=func, nvar=nvar, n=1, dx=dx, order=order )
    rst = [ ]
    for i in range( nvar ):
        rst.append( gradient( grad[ i ], nvar=nvar, n=1, dx=dx, order=order ) )
    return rst

def gramSchmid( A, alighVec=None, reverse=False ):

    A = np.array( A, dtype=float )
    # Check edge case
    if A.ndim != 2:
        raise ValueError( "A should be a 2d matrix." )

    if A.shape[ 0 ] != A.shape[ 1 ]:
        raise ValueError( "A should be a square matrix." )

    dim = A.shape[ 0 ]
    if alighVec is None:
        alighVec = A[ :, 0 ]
    B = np.copy( A )
    # Check if alighVec coincides with one direction in A
    for i in range( 1, dim ):
        curVec = A[ :, i ]
        normCurVec = np.linalg.norm( curVec )
        normAlignVec = np.linalg.norm( alighVec )
        if np.allclose( np.dot( curVec, alighVec ), normCurVec * normAlignVec ):
            B[ :, : dim - i ] = np.copy( A[ :, i: ] )
            B[ :, i: ] = np.copy( A[ :, : dim - i ])
    
    B[ :, 0 ] = alighVec / np.linalg.norm( alighVec )

    for i in range( 1, dim ):
        curVec = np.copy( B[ :, i ] )
        for j in range( 0, i ):
            projVec = B[ :, j ]
            curVec = curVec - np.dot( projVec, curVec ) / \
                np.dot( projVec, projVec ) * projVec
        
        B[ :, i ] = np.copy( curVec / np.linalg.norm( curVec ) )
    
    if reverse:
        B = np.fliplr( B )
    
    J = np.linalg.solve( A.T, B.T ).T
    return B, J
