#!/usr/bin/env python3

'''
This module implements the mean stress methods to quantify the interaction of mean 
and alternating stresses on the fatigue life of a material.
'''

import numpy as np


def goodmanCorrection( stressRange, ultimateStrength, n=1.0 ):
    '''
    The Goodman correction in this implementation is applicable to cases with stress 
    ratio no less than -1.

    Parameters
    ----------
    stressRange: 1d array
        Stress range, e.g., [ lowerStress, upperStress ].
    ultimateStrength: scalar
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
        If ultimateStrength is not a scalar or ultimateStrength <= 0.
        If ultimateStrength is smaller than upper stress stressRange[ 1 ].
        If n < 1.0.

    Examples
    --------
    >>> from ffpack.lcc import goodmanCorrection
    >>> stressRange = [ 1.0, 2.0 ]
    >>> ultimateStrength = 4.0
    >>> rst = goodmanCorrection( stressRange, ultimateStrength )
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
    if not isinstance( ultimateStrength, int ) and not isinstance( ultimateStrength, float ):
        raise ValueError( "ultimateStrength should be a scalar" )
    if ultimateStrength <= 0:
        raise ValueError( "ultimateStrength should be positive" )
    if ultimateStrength < stressRange[ 1 ]:
        raise ValueError( "ultimateStrength should not be smaller than uppder stress" )

    stressRatio = stressRange[ 0 ] / stressRange[ 1 ]
    if stressRatio < -1:
        raise ValueError( "Stress ratio should be no less than -1" )
    
    if not isinstance( n, int ) and not isinstance( n, float ):
        raise ValueError( "n should be a scalar" )
    if n < 1.0:
        raise ValueError( "Safety factor should be no less than 1.0" )
    
    sigmaMean = ( stressRange[ 0 ] + stressRange[ 1 ] ) / 2.0
    sigmaAlt = ( stressRange[ 1 ] - stressRange[ 0 ] ) / 2.0

    rst = sigmaAlt / ( n - sigmaMean / ultimateStrength )

    return rst


def soderbergCorrection( stressRange, yieldStrength, n=1.0 ):
    '''
    The Soderberg correction in this implementation is applicable to cases with stress 
    ratio no less than -1.

    Parameters
    ----------
    stressRange: 1d array
        Stress range, e.g., [ lowerStress, upperStress ].
    yieldStrength: scalar
        Yield strength.
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
        If yieldStrength is not a scalar or yieldStrength <= 0.
        If yieldStrength is smaller than upper stress stressRange[ 1 ].
        If safety factor n < 1.0.

    Examples
    --------
    >>> from ffpack.lcc import soderbergCorrection
    >>> stressRange = [ 1.0, 2.0 ]
    >>> yieldStrength = 3.0
    >>> rst = soderbergCorrection( stressRange, yieldStrength )
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
    if not isinstance( yieldStrength, int ) and not isinstance( yieldStrength, float ):
        raise ValueError( "yieldStrength should be a scalar" )
    if yieldStrength <= 0:
        raise ValueError( "yieldStrength should be positive" )
    if yieldStrength < stressRange[ 1 ]:
        raise ValueError( "yieldStrength should not be smaller than uppder stress" )

    stressRatio = stressRange[ 0 ] / stressRange[ 1 ]
    if stressRatio < -1:
        raise ValueError( "Stress ratio should be no less than -1" )
    
    if not isinstance( n, int ) and not isinstance( n, float ):
        raise ValueError( "n should be a scalar" )
    if n < 1.0:
        raise ValueError( "Safety factor should be no less than 1.0" )
    
    sigmaMean = ( stressRange[ 0 ] + stressRange[ 1 ] ) / 2.0
    sigmaAlt = ( stressRange[ 1 ] - stressRange[ 0 ] ) / 2.0

    rst = sigmaAlt / ( n - sigmaMean / yieldStrength )

    return rst


def gerberCorrection( stressRange, yieldStrength, n=1.0 ):
    '''
    The Gerber correction in this implementation is applicable to cases with stress 
    ratio no less than -1.

    Parameters
    ----------
    stressRange: 1d array
        Stress range, e.g., [ lowerStress, upperStress ].
    yieldStrength: scalar
        Yield strength.
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
        If yieldStrength is not a scalar or yieldStrength <= 0.
        If yieldStrength is smaller than upper stress stressRange[ 1 ].
        If safety factor n < 1.0.

    Examples
    --------
    >>> from ffpack.lcc import gerberCorrection
    >>> stressRange = [ 1.0, 2.0 ]
    >>> yieldStrength = 3.0
    >>> rst = gerberCorrection( stressRange, yieldStrength )
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
    if not isinstance( yieldStrength, int ) and not isinstance( yieldStrength, float ):
        raise ValueError( "yieldStrength should be a scalar" )
    if yieldStrength <= 0:
        raise ValueError( "yieldStrength should be positive" )
    if yieldStrength < stressRange[ 1 ]:
        raise ValueError( "yieldStrength should not be smaller than uppder stress" )

    stressRatio = stressRange[ 0 ] / stressRange[ 1 ]
    if stressRatio < -1:
        raise ValueError( "Stress ratio should be no less than -1" )
    
    if not isinstance( n, int ) and not isinstance( n, float ):
        raise ValueError( "n should be a scalar" )
    if n < 1.0:
        raise ValueError( "Safety factor should be no less than 1.0" )
    
    sigmaMean = ( stressRange[ 0 ] + stressRange[ 1 ] ) / 2.0
    sigmaAlt = ( stressRange[ 1 ] - stressRange[ 0 ] ) / 2.0

    rst = 1 - ( n * sigmaMean / yieldStrength) ** 2
    rst = n * sigmaAlt / rst

    return rst
