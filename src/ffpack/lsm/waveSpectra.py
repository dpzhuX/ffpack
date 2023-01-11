#!/usr/bin/env python3

import numpy as np
from scipy import special


def piersonMoskowitzSpectrum( w, Uw, alpha=0.0081, beta=0.74, g=9.81 ):
    '''
    Pierson Moskowitz spectrum is an empirical relationship 
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
    >>> rst = piersonMoskowitzSpectrum( w, Uw, alpha=0.0081, 
    ...                                 beta=1.25, gamma=3.3, g=9.81 )
    '''
    if not isinstance( w, int ) and not isinstance( w, float ):
        raise ValueError( "w should be a scalar" )
    if not isinstance( Uw, int ) and not isinstance( Uw, float ):
        raise ValueError( "Uw should be a scalar" )

    rst = alpha * g * g / np.power( w, 5 ) * np.exp( -beta * 
                                                     np.power( ( g / Uw ) / w, 4 ) )
    return rst



def jonswapSpectrum( w, wp, alpha=0.0081, beta=1.25, gamma=3.3, g=9.81 ):
    '''
    JONSWAP (Joint North Sea Wave Project) spectrum is an empirical relationship 
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
        raise ValueError( "wp should be a scalar" )

    sigma = 0.07 
    if ( w > wp ):
        sigma = 0.09
    r = np.exp( -( w - wp ) * ( w - wp ) / ( 2 * wp * wp * sigma * sigma ) )
    rst = alpha * g * g / np.power( w, 5 ) * \
        np.exp( -beta * np.power( wp / w, 4 ) ) * np.power( gamma, r )
    return rst



def isscSpectrum( w, wp, Hs ):
    '''
    ISSC spectrum, also known as Bretschneider or modified Pierson-Moskowitz. 

    Parameters
    ----------
    w: scalar
        Wave frequency.
    wp: scalar
        Peak wave frequency.
    Hs: scalar
        Significant wave height.
    
    Returns
    -------
    rst: scalar
        The wave spectrum density value at wave frequency w.
    
    Raises
    ------
    ValueError
        If w is not a scalar.
        If wp is not a scalar.
        If Hs is not a scalar.

    Examples
    --------
    >>> from ffpack.lsm import isscSpectrum
    >>> w = 0.02
    >>> wp = 0.51
    >>> Hs = 20
    >>> rst = isscSpectrum( w, wp, Hs )
    '''
    if not isinstance( w, int ) and not isinstance( w, float ):
        raise ValueError( "w should be a scalar" )
    if not isinstance( wp, int ) and not isinstance( wp, float ):
        raise ValueError( "wp should be a scalar" )
    if not isinstance( Hs, int ) and not isinstance( Hs, float ):
        raise ValueError( "Hs should be a scalar" )
    
    wwp4 = np.power( wp / w, 4 )
    rst = 5 / 16 * Hs * Hs * wwp4 / w * np.exp( -1.25 * wwp4 )
    return rst



def gaussianSwellSpectrum( w, wp, Hs, sigma ):
    '''
    Gaussian Swell spectrum, typically used to model long period 
    swell seas [Guidance2016A]_. 

    Parameters
    ----------
    w: scalar
        Wave frequency.
    wp: scalar
        Peak wave frequency.
    Hs: scalar
        Significant wave height.
    sigma: scalar
        peakedness parameter for Gaussian spectral width.
    
    Returns
    -------
    rst: scalar
        The wave spectrum density value at wave frequency w.
    
    Raises
    ------
    ValueError
        If w is not a scalar.
        If wp is not a scalar.
        If Hs is not a scalar.
        If sigma is not a scalar.

    Examples
    --------
    >>> from ffpack.lsm import gaussianSwellSpectrum
    >>> w = 0.02
    >>> wp = 0.51
    >>> Hs = 20
    >>> sigma = 0.07
    >>> rst = gaussianSwellSpectrum( w, wp, Hs, sigma )

    References
    ----------
    .. [Guidance2016A] Guidance Notes on Selecting Design Wave by Long 
       Term Stochastic Method
    '''
    if not isinstance( w, int ) and not isinstance( w, float ):
        raise ValueError( "w should be a scalar" )
    if not isinstance( wp, int ) and not isinstance( wp, float ):
        raise ValueError( "wp should be a scalar" )
    if not isinstance( Hs, int ) and not isinstance( Hs, float ):
        raise ValueError( "Hs should be a scalar" )
    if not isinstance( sigma, int ) and not isinstance( sigma, float ):
        raise ValueError( "sigma should be a scalar" )
    
    twoPi = 2 * np.pi
    pexp = np.power( ( w - wp ) / ( twoPi * sigma ), 2 ) / 2
    rst = Hs * Hs / ( 16 * sigma * np.power( twoPi, 1.5 ) ) * np.exp( -pexp )
    return rst



def ochiHubbleSpectrum( w, wp1, wp2, Hs1, Hs2, lambda1, lambda2 ):
    '''
    Ochi-Hubble spectrum covers shapes of wave spectra associated with the growth 
    and decay of a storm, including swells. [Guidance2016B]_. 

    Parameters
    ----------
    w: scalar
        Wave frequency.
    wp1, wp2: scalar
        Peak wave frequency.
    Hs1, Hs2: scalar
        Significant wave height.
    lambda1, lambda2: scalar
    
    Returns
    -------
    rst: scalar
        The wave spectrum density value at wave frequency w.
    
    Raises
    ------
    ValueError
        If w is not a scalar.
        If wp1 or wp2 is not a scalar.
        If Hs1 or Hs2 is not a scalar.
        If lambda1 or lambda2 is not a scalar.
        If wp1 is not smaller than wp2.

    Notes
    -----
    Hs1 should normally be greater than Hs2 since most of the wave energy tends to 
    be associated with the lower frequency component.

    Examples
    --------
    >>> from ffpack.lsm import ochiHubbleSpectrum
    >>> w = 0.02
    >>> wp1 = 0.4
    >>> wp2 = 0.51
    >>> Hs1 = 20
    >>> Hs2 = 15
    >>> lambda1 = 7
    >>> lambda2 = 10
    >>> rst = ochiHubbleSpectrum( w, wp1, wp2, Hs1, Hs2, lambda1, lambda2 )

    References
    ----------
    .. [Guidance2016B] Guidance Notes on Selecting Design Wave by Long 
       Term Stochastic Method
    '''
    if not isinstance( w, int ) and not isinstance( w, float ):
        raise ValueError( "w should be a scalar" )
    if not isinstance( wp1, int ) and not isinstance( wp1, float ):
        raise ValueError( "wp1 should be a scalar" )
    if not isinstance( wp2, int ) and not isinstance( wp2, float ):
        raise ValueError( "wp2 should be a scalar" )
    if not isinstance( Hs1, int ) and not isinstance( Hs1, float ):
        raise ValueError( "Hs1 should be a scalar" )
    if not isinstance( Hs2, int ) and not isinstance( Hs2, float ):
        raise ValueError( "Hs2 should be a scalar" )
    if not isinstance( lambda1, int ) and not isinstance( lambda1, float ):
        raise ValueError( "lambda1 should be a scalar" )
    if not isinstance( lambda2, int ) and not isinstance( lambda2, float ):
        raise ValueError( "lambda2 should be a scalar" )
    if wp1 >= wp2:
        raise ValueError( "wp1 must be less than wp2" )

    
    def oneTerm( w, wp, Hs, lambdaVal ):
        fourLambda = ( 4 * lambdaVal + 1 ) / 4
        firstPart = np.power( fourLambda * np.power( wp, 4 ), lambdaVal ) / \
            special.gamma( lambdaVal )
        expc = np.exp( -fourLambda * np.power( wp / w, 4 ) )
        rst = firstPart * Hs * Hs / np.power( w, fourLambda * 4 ) * expc
        return rst
    
    rst = ( oneTerm( w, wp1, Hs1, lambda1 ) + oneTerm( w, wp2, Hs2, lambda2 ) ) / 4
    return rst
