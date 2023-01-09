#!/usr/bin/env python3

import numpy as np


def davenportSpectrumWithDragCoef( n, delta1, kappa=0.005, normalized=True ):
    '''
    Davenport spectrum in the original paper by Davenport [1]_.

    Parameters
    ----------
    n: scalar
        Frequency ( Hz ).
    delta1: scalar
        Velocity ( m/s ) at standard reference height of 10 m.
    kappa: scalar, optional
        Drag coefficient referred to mean velocity at 10 m.
        Default value 0.005 corresponding to open unobstructed country [1]_.
        The recommended value for heavilly built-up urban centers with tall buildings is 0.05.
        The recommended value for country broken by low clustered obstructions is between 0.015 and 0.02. 
    normalized: bool, optional
        If normalized is set to False, the power spectrum density will be returned.
    
    Returns
    -------
    rst: scalar
        Power spectrum density ( m^2 s^-2 Hz^-1 ) when normalized=False.
        Normalized power spectrum density when normalized=True.
    
    Raises
    ------
    ValueError
        If n is not a scalar.
        If delta1 is not a scalar.

    Examples
    --------
    >>> from ffpack.lsm import davenportSpectrumWithDragCoef
    >>> n = 2
    >>> delta1 = 10
    >>> rst = davenportSpectrumWithDragCoef( n, delta1, kappa=0.005, normalized=True )

    References
    ----------
    .. [1] Davenport, A. G. (1961). The spectrum of horizontal gustiness near the ground in high winds. 
           Quarterly Journal of the Royal Meteorological Society, 87(372), 194-211.
    '''
    if not isinstance( n, int ) and not isinstance( n, float ):
        raise ValueError( "w should be a scalar" )
    if not isinstance( delta1, int ) and not isinstance( delta1, float ):
        raise ValueError( "Uw should be a scalar" )

    x = 1200 * n / delta1
    rst = 4.0 * x * x / np.power( 1 + x * x, 4 / 3 )
    if normalized:
        return rst
    
    rst = rst * kappa * delta1 * delta1 / n
    return rst



def davenportSpectrumWithRoughnessLength( n, uz, z=10, z0=0.03, normalized=True ):
    '''
    Davenport spectrum in the paper by Hiriart et al. [1]_.

    Parameters
    ----------
    n: scalar
        Frequency ( Hz ).
    uz: scalar
        Mean wind speed ( m/s ) measured at height z.
    z: scalar
        Height above the ground ( m ).
    z0: scalar, optional
        Roughness length to mean velocity at 10 m.
        Default value 0.03 corresponding to open in NIST database [2]_.
        The recommended value for heavilly built-up urban centers with tall buildings is 0.05.
        The recommended value for country broken by low clustered obstructions is between 0.015 and 0.02. 
    normalized: bool, optional
        If normalized is set to False, the power spectrum density will be returned.
    
    Returns
    -------
    rst: scalar
        Power spectrum density ( m^2 s^-2 Hz^-1 ) when normalized=False.
        Normalized power spectrum density when normalized=True.
    
    Raises
    ------
    ValueError
        If n is not a scalar.
        If delta1 is not a scalar.

    Examples
    --------
    >>> from ffpack.lsm import davenportSpectrumWithRoughnessLength
    >>> n = 2
    >>> delta1 = 10
    >>> rst = davenportSpectrumWithRoughnessLength( n, delta1, kappa=0.005, normalized=True )

    References
    ----------
    .. [1] Hiriart, D., Ochoa, J. L., & Garcia, B. (2001). Wind power spectrum measured at 
           the San Pedro Mártir Sierra. Revista Mexicana de Astronomia y Astrofisica, 37(2), 213-220.
    .. [2] Ho, T. C. E., Surry, D., & Morrish, D. P. (2003). NIST/TTU cooperative agreement-windstorm 
           mitigation initiative: Wind tunnel experiments on generic low buildings. London, Canada: 
           BLWTSS20-2003, Boundary-Layer Wind Tunnel Laboratory, Univ. of Western Ontario.
    '''
    if not isinstance( n, int ) and not isinstance( n, float ):
        raise ValueError( "w should be a scalar" )
    if not isinstance( uz, int ) and not isinstance( uz, float ):
        raise ValueError( "uz should be a scalar" )

    x = 1200 * n / uz
    rst = 4.0 * x * x / np.power( 1 + x * x, 4 / 3 )
    if normalized:
        return rst
    
    uf = 0.4 * uz / np.log( z / z0 )
    rst = rst * uf * uf / n
    return rst
