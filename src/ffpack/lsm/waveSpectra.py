#!/usr/bin/env python3

import numpy as np


def piersonMoskowitzSpectrum( w, Uw, alpha=0.0081, beta=0.74, g=9.81 ):
    '''
    Pierson Moskowitz spectra is an empirical relationship 
    that defines the distribution of energy with frequency within the ocean.

    Parameters
    ----------
    w: scalar
        Wave frequency.
    Uw: scalar
        Wind speed at a height of 19.5m above the sea surface.
    alpha: scalar, optional
        Intensity of the Spectra.
    beta: scalar, optional
        Shape factor.
    g: scalar, optional
        Acceleration due to gravity, a constant.
        9.81 m/s2 in SI units.
    
    Returns
    -------
    rst: scalar
        The wave spectrum density value at wave frequency w.
    
    Raises
    ------
    ValueError
        If w is not a scalar.
        If wp is not a scalar.

    Examples
    --------
    >>> from ffpack.lsm import piersonMoskowitzSpectrum
    >>> w = 0.51
    >>> Uw = 20
    >>> rst = piersonMoskowitzSpectrum( w, Uw, alpha=0.0081, beta=1.25, gamma=3.3, g=9.81 )
    '''
    if not isinstance( w, int ) and not isinstance( w, float ):
        raise ValueError( "w should be a scalar" )
    if not isinstance( Uw, int ) and not isinstance( Uw, float ):
        raise ValueError( "Uw should be a scalar")

    rst = alpha * g * g / np.power( w, 5 ) * np.exp( -beta * 
                                                     np.power( ( g / Uw ) / w, 4 ) )
    return rst



def jonswapSpectrum( w, wp, alpha=0.0081, beta=1.25, gamma=3.3, g=9.81 ):
    '''
    JONSWAP (Joint North Sea Wave Project) spectra is an empirical relationship 
    that defines the distribution of energy with frequency within the ocean.

    Parameters
    ----------
    w: scalar
        Wave frequency.
    wp: scalar
        Peak wave frequency.
    alpha: scalar, optional
        Intensity of the Spectra.
    beta: scalar, optional
        Shape factor, fixed value 1.25.
    gamma: scalar, optional
        Peak enhancement factor.
    g: scalar, optional
        Acceleration due to gravity, a constant.
        9.81 m/s2 in SI units.
    
    Returns
    -------
    rst: scalar
        The wave spectrum density value at wave frequency w.
    
    Raises
    ------
    ValueError
        If w is not a scalar.
        If wp is not a scalar.

    Examples
    --------
    >>> from ffpack.lsm import jonswapSpectrum
    >>> w = 0.02
    >>> wp = 0.51
    >>> rst = jonswapSpectrum( w, wp, alpha=0.0081, beta=1.25, gamma=3.3, g=9.81 )
    '''
    if not isinstance( w, int ) and not isinstance( w, float ):
        raise ValueError( "w should be a scalar" )
    if not isinstance( wp, int ) and not isinstance( wp, float ):
        raise ValueError( "wp should be a scalar")

    sigma = 0.07 
    if ( w > wp ):
        sigma = 0.09
    r = np.exp( -( w - wp ) * ( w - wp ) / ( 2 * wp * wp * sigma * sigma ) )
    rst = alpha * g * g / np.power( w, 5 ) * np.exp( -beta * np.power( wp / w, 4 ) ) * np.power( gamma, r )
    return rst
