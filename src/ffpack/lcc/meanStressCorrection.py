#!/usr/bin/env python3

'''
This module implements the mean stress methods to quantify the interaction of mean 
and alternating stresses on the fatigue life of a material.
'''

import numpy as np


def goodmanCorrection( stressRange, sigma, n=1.0 ):
    '''
    The Goodman correction in this implementation is applicable to cases with stress 
    ratio no less than -1

    Parameters
    ----------
    stressRange: 1d array
        Stress range, e.g., [ lowerStress, upperStress ].
    sigma: scalar
        Ultimate tensile strength.
    n: scalar, optional
        Safety factor, default to 1.0.
    
    Returns
    -------
    rst: scalar
        Fatigue limit.
    
    Raises
    ------
    ValueError
        If the stressRange dimension is not 1, or stressRange length is not 2.
        If stressRange[ 1 ] <= 0 or stressRange[ 0 ] >= stressRange[ 1 ].
        If sigma is not a scalar or sigma <= 0.
        If sigma is smaller than upper stress stressRange[ 1 ].
        If n < 1.0.

    Examples
    --------
    >>> from ffpack.lcc import goodmanCorrection
    >>> stressRange = [ 1.0, 2.0 ]
    >>> sigma = 4.0
    >>> rst = goodmanCorrection( stressRange, sigma )
    
    References
    ----------
    .. [Lee2011] Lee, Y.L., Barkey, M.E. and Kang, H.T., 2011. Metal fatigue analysis 
       handbook: practical problem-solving techniques for computer-aided engineering. 
       Elsevier.
    '''
    stressRange = np.array( stressRange )
    if len( stressRange.shape ) != 1:
        raise ValueError( "Input stressRange dimension should be 1" )
    if stressRange.shape[ 0 ] != 2:
        raise ValueError( "Input stressRange length should be 2" )
    if stressRange[ 1 ] <= 0:
        raise ValueError( "Input stressRange should have upper stress stressRange[ 1 ] > 0" )
    if stressRange[ 1 ] <= stressRange[ 0 ]:
        raise ValueError( 
            "Input stressRange should have lower stress stressRange[ 0 ] < upper stress stressRange[ 1 ]" )
    if not isinstance( sigma, int ) and not isinstance( sigma, float ):
        raise ValueError( "sigma should be a scalar" )
    if sigma <= 0:
        raise ValueError( "sigma should be positive" )
    if sigma < stressRange[ 1 ]:
        raise ValueError( "sigma should not be smaller than uppder stress" )

    stressRatio = stressRange[ 0 ] / stressRange[ 1 ]
    if stressRatio < -1:
        raise ValueError( "Stress ratio should be no less than -1" )
    
    if not isinstance( n, int ) and not isinstance( n, float ):
        raise ValueError( "n should be a scalar" )
    if n < 1.0:
        raise ValueError( "Safety factor should be no less than 1.0" )
    
    sigmaMean = ( stressRange[ 0 ] + stressRange[ 1 ] ) / 2.0
    sigmaAlt = ( stressRange[ 1 ] - stressRange[ 0 ] ) / 2.0

    rst = sigmaAlt / ( n - sigmaMean / sigma )

    return rst
