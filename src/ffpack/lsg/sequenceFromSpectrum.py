#!/usr/bin/env python3

import numpy as np


def harmonicSuperposition( fs, time, freq, psd, freqBandwidth=None ):
    '''
    Generate load sequence by a random walk.

    Parameters
    ----------
    fs: scalar 
        Sampling frequence.
    time: scalar
        Total sampling time.
    freq: 1darray
        Frequence array for psd
    psd: 1darray
        Power spectrum density array. 
    freqBandwidth: scalar, optional
        Frequence bandwidth used to generate the time series from psd.
        Default to None, every frequence in freq will be used. 
    
    Returns
    -------
    ts: 1darray
        Array containing all the time data for the time series.
    amp: 1darray
        Amplitude array containing the amplitudes of the time series 
        corresponding to ts.

    
    Raises
    ------
    ValueError
        If the numSteps is less than 1 or the dim is less than 1.

    Examples
    --------
    >>> from ffpack.lsg import harmonicSuperposition
    >>> fs = 100
    >>> time = 10
    >>> freq = [ 0, 0.1, 0.2, 0.3, 0.4, 0.5 ]
    >>> psd = [ 0.01, 2, 0.05, 0.04, 0.01, 0.03 ]
    >>> ts, amp = harmonicSuperposition( fs, time, freq, psd, freqBandwidth=None )

    '''
    # edge case check for fs and time
    if not isinstance( fs, int ) and not isinstance( fs, float ):
        raise ValueError( "fs should be a scalar" )
    if fs <= 0:
        raise ValueError( "fs should be positive" )
    if not isinstance( time, int ) and not isinstance( time, float ):
        raise ValueError( "time should be a scalar" )
    if time <= 0:
        raise ValueError( "time should be positive" )
    
    # edge case check for freq and psd
    freq = np.array( freq, dtype=float )
    if len( freq.shape ) != 1:
        raise ValueError( "freq dimension should be 1" )
    if freq.shape[ 0 ] < 2:
        raise ValueError( "freq length should be at least 2" )
    psd = np.array( psd, dtype=float )
    if len( psd.shape ) != 1:
        raise ValueError( "psd dimension should be 1" )
    if psd.shape[ 0 ] < 2:
        raise ValueError( "psd length should be at least 2" )
    if len( freq ) != len( psd ):
        raise ValueError( "freq and psd should be in the same length" )
    if freq[ 0 ] < 0:
        raise ValueError( " freq array should be non-negative" )
    for i in range( 1, len( freq ) ):
        if freq[ i ] <= freq[ i - 1 ]:
            raise ValueError( " freq array should be strictly increasing" )

    # edge case check for freqBandwidth
    if freqBandwidth is not None:
        if not isinstance( freqBandwidth, int ) and not isinstance( freqBandwidth, float ):
            raise ValueError( "freqBandwidth should be a scalar" )
        if freqBandwidth <= 0:
            raise ValueError( "freqBandwidth should be positive" )

    n = round( fs * time )
    ts = 1 / fs * np.arange( n, dtype=float )
    amp = np.zeros( n )
    # generate phase angle
    phis = -np.pi + 2 * np.pi * np.random.randn( len( freq ) )
    # deal with freqBandwidth
    if freqBandwidth is None:
        freqBandwidth = freq[ 1 ] - freq[ 0 ]

    for i in range( len( freq ) ):
        for j in range( n ):
            amp[ j ] += np.sqrt( 2 * psd[ i ] * freqBandwidth ) * \
                np.sin( 2 * np.pi * freq[ i ] * ts[ j ] + phis[ i ] )

    return ts, amp
