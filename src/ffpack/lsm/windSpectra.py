#!/usr/bin/env python3

import numpy as np


def davenportSpectrumWithDragCoef( n, delta1, kappa=0.005, normalized=True ):
    '''
    Davenport spectrum in the original paper by Davenport [Davenport1961]_.

    Parameters
    ----------
    n: scalar
        Frequency ( Hz ) when normalized=False.
        Normalized frequency when normalized=True.
    delta1: scalar
        Velocity ( m/s ) at standard reference height of 10 m.
    kappa: scalar, optional
        Drag coefficient referred to mean velocity at 10 m.  Default value 0.005 
        corresponding to open unobstructed country [Davenport1961]_.
        The recommended value for heavilly built-up urban centers with 
        tall buildings is 0.05. The recommended value for country broken by 
        low clustered obstructions is between 0.015 and 0.02. 
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
    >>> rst = davenportSpectrumWithDragCoef( n, delta1, kappa=0.005, 
    ...                                      normalized=True )

    References
    ----------
    .. [Davenport1961] Davenport, A.G., 1961. The spectrum of horizontal gustiness 
       near the ground in high winds. Quarterly Journal of the Royal Meteorological 
       Society, 87(372), pp.194-211.
    '''
    if not isinstance( n, int ) and not isinstance( n, float ):
        raise ValueError( "n should be a scalar" )
    if not isinstance( delta1, int ) and not isinstance( delta1, float ):
        raise ValueError( "delta1 should be a scalar" )

    def rightPart( x ):
        rst = 4.0 * x * x / np.power( 1 + x * x, 4 / 3 )
        return rst

    if normalized:
        x = 120 * n
        return rightPart( x )
    
    x = 1200 * n / delta1
    rst = rightPart( x ) * kappa * delta1 * delta1 / n
    return rst



def davenportSpectrumWithRoughnessLength( n, uz, z=10, z0=0.03, normalized=True ):
    '''
    Davenport spectrum in the paper by Hiriart et al. [Hiriart2001]_.

    Parameters
    ----------
    n: scalar
        Frequency ( Hz ) when normalized=False.
        Normalized frequency when normalized=True.
    uz: scalar
        Mean wind speed ( m/s ) measured at height z.
    z: scalar, optional
        Height above the ground ( m ), default to 10 m. 
    z0: scalar, optional
        Roughness length ( m ), default to 0.03 m corresponding to open 
        exposure case in [Ho2003]_.
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
        If uz is not a scalar.

    Examples
    --------
    >>> from ffpack.lsm import davenportSpectrumWithRoughnessLength
    >>> n = 2
    >>> uz = 10
    >>> rst = davenportSpectrumWithRoughnessLength( n, uz, z=10, z0=0.03, 
    ...                                             normalized=True )

    References
    ----------
    .. [Hiriart2001] Hiriart, D., Ochoa, J.L. and Garcia, B., 2001. Wind power 
       spectrum measured at the San Pedro Mártir Sierra. Revista Mexicana de 
       Astronomia y Astrofisica, 37(2), pp.213-220.
    .. [Ho2003] Ho, T.C.E., Surry, D. and Morrish, D.P., 2003. NIST/TTU cooperative 
       agreement-windstorm mitigation initiative: Wind tunnel experiments on generic 
       low buildings. London, Canada: BLWTSS20-2003, Boundary-Layer Wind Tunnel 
       Laboratory, Univ. of Western Ontario.
    '''
    if not isinstance( n, int ) and not isinstance( n, float ):
        raise ValueError( "n should be a scalar" )
    if not isinstance( uz, int ) and not isinstance( uz, float ):
        raise ValueError( "uz should be a scalar" )

    def rightPart( x ):
        rst = 4.0 * x * x / np.power( 1 + x * x, 4 / 3 )
        return rst
    
    if normalized:
        x = 1200 / z * n
        return rightPart( x )
    
    x = 1200 * n / uz
    uf = 0.4 * uz / np.log( z / z0 )
    rst = rightPart( x ) * uf * uf / n
    return rst



def ec1Spectrum( n, uz, sigma=0.03, z=10, tcat=0, normalized=True ):
    '''
    EC1 spectrum is implemented according to Annex B [EN1991-1-42005]_.

    Parameters
    ----------
    n: scalar
        Frequency ( Hz ) when normalized=False.
        Normalized frequency when normalized=True.
    uz: scalar
        Mean wind speed ( m/s ) measured at height z.
    sigma: scalar, optional
        Standard derivation of wind.  
    z: scalar, optional
        Height above the ground ( m ), default to 10 m. 
    tcat: scalar, optional
        Terrain category, could be 0, 1, 2, 3, 4
        Default to 0 (sea or coastal area exposed to the open sea) in EC1 Table 4.1.
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
        If uz is not a scalar.
        If tcat is not int or not within range of 0 to 4

    Examples
    --------
    >>> from ffpack.lsm import ec1Spectrum
    >>> n = 2
    >>> uz = 10
    >>> rst = ec1Spectrum( n, uz, sigma=0.03, z=10, tcat=0, normalized=True )

    References
    ----------
    .. [EN1991-1-42005] EN1991-1-4, 2005. Eurocode 1: Actions on structures.
    '''
    if not isinstance( n, int ) and not isinstance( n, float ):
        raise ValueError( "n should be a scalar" )
    if not isinstance( uz, int ) and not isinstance( uz, float ):
        raise ValueError( "uz should be a scalar" )
    if not isinstance( tcat, int ):
        raise ValueError( "tcat should be an integer" )
    if tcat < 0 or tcat > 4:
        raise ValueError( "tcat could only be 0, 1, 2, 3, or 4" )

    def rightPart( x ):
        rst = 6.8 * x / np.power( 1 + 10.2 * x, 5 / 3 )
        return rst
    
    if normalized:
        return rightPart( n )
    
    options = { 0: [ 0.003, 1 ], 
                1: [ 0.01, 1 ],
                2: [ 0.05, 2 ],
                3: [ 0.3, 5 ],
                4: [ 1.0, 10 ] }
    [ z0, zmin ] = options[ tcat ]

    alpha = 0.67 + 0.05 * np.log( z0 )
    lz = 300 * np.power( max( z, zmin ) / 200, alpha )
    f = n * lz / uz
    rst = rightPart( f ) * sigma * sigma / n
    return rst



def iecSpectrum( f, vhub, sigma=0.03, z=10, k=1, normalized=True ):
    '''
    IEC spectrum is implemented according to [IEC2005]_.

    Parameters
    ----------
    f: scalar
        Frequency ( Hz ) when normalized=False.
        Normalized frequency when normalized=True.
    vhub: scalar
        Mean wind speed ( m/s ).
    sigma: scalar, optional
        Standard derivation of the turblent wind speed component.  
    z: scalar, optional
        Height above the ground ( m ), default to 10 m. 
    k: scalar, optional
        Wind speed direction, could be 1, 2, 3
        ( 1 = longitudinal, 2 = lateral, and 3 = upward )
        Default to 1 (longitudinal).
    normalized: bool, optional
        If normalized is set to False, the power spectrum density will be returned.
    
    Returns
    -------
    rst: scalar
        Single-sided velocity component power spectrum density ( m^2 s^-2 Hz^-1 ) 
        when normalized=False.
        Normalized single-sided velocity component power spectrum density 
        when normalized=True.
    
    Raises
    ------
    ValueError
        If n is not a scalar.
        If uz is not a scalar.
        If k is not int or not within range of 1 to 3

    Examples
    --------
    >>> from ffpack.lsm import iecSpectrum
    >>> n = 2
    >>> vhub = 10
    >>> rst = iecSpectrum( n, vhub, sigma=0.03, z=10, k=1, normalized=True )

    References
    ----------
    .. [IEC2005] IEC, 2005. IEC 61400-1, Wind turbines - Part 1: Design requirements.
    '''
    if not isinstance( f, int ) and not isinstance( f, float ):
        raise ValueError( "f should be a scalar" )
    if not isinstance( vhub, int ) and not isinstance( vhub, float ):
        raise ValueError( "vhub should be a scalar" )
    if not isinstance( k, int ):
        raise ValueError( "k should be an integer" )
    if k < 1 or k > 3:
        raise ValueError( "k could only be 1, 2, or 3" )

    def rightPart( f ):
        rst = 4 * f / np.power( 1 + 6 * f, 5 / 3 ) 
        return rst 

    if normalized:
        return rightPart( f )

    lambda1 = 0.7 * z
    if z >= 60:
        lambda1 = 42
    
    factors = { 1: [ 1, 8.1 ],
                2: [ 0.8, 2.7 ],
                3: [ 0.5, 0.66 ] }[ k ]
    sigmak = factors[ 0 ] * sigma
    lk = factors[ 1 ] * lambda1
    nf = f * lk / vhub
    rst = rightPart( nf ) * sigmak * sigmak / f
    return rst



def apiSpectrum( f, u0, z=10 ):
    '''
    API spectrum is implemented according to [API2007]_.

    Parameters
    ----------
    f: scalar
        Frequency ( Hz ).
    u0: scalar
        1 hour mean wind speed ( m/s ) at 10 m above sea level.
    
    Returns
    -------
    rst: scalar
        Power spectrum density ( m^2 s^-2 Hz^-1 ).
    
    Raises
    ------
    ValueError
        If n is not a scalar.
        If uz is not a scalar.

    Examples
    --------
    >>> from ffpack.lsm import apiSpectrum
    >>> f = 2
    >>> u0 = 10
    >>> rst = apiSpectrum( f, u0 )

    References
    ----------
    .. [API2007] API, 2007. Recommended practice 2A-WSD (RP 2A-WSD): 
       Recommnded practice for planning, designing and constructing fixed offshore 
       platforms - working stress design.
    '''
    if not isinstance( f, int ) and not isinstance( f, float ):
        raise ValueError( "f should be a scalar" )
    if not isinstance( u0, int ) and not isinstance( u0, float ):
        raise ValueError( "u0 should be a scalar" )

    n = 0.468
    ftilde = 172 * f * np.power( z / 10, 2 / 3 ) * np.power( u0 / 10, -0.75 )
    rst = 320 * np.power( u0 / 10, 2 ) * np.power( z / 10, 0.45 ) 
    rst = rst / np.power( 1 + np.power( ftilde, n ), 5 / ( 3 * n ) )
    return rst
