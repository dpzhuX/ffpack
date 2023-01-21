#!/usr/bin/env python3

import numpy as np
from ffpack.config import globalConfig


def harmonicSuperposition( fs, time, freq, psd, freqBandwidth=None ):
    '''
    Generate a sequence from a given power spectrum density with harmonic 
    superposition method.

    Parameters
    ----------
    fs: scalar 
        Sampling frequency.
    time: scalar
        Total sampling time.
    freq: 1darray
        Frequency array for psd.
        The freq array should be in equally spaced increasing. 
    psd: 1darray
        Power spectrum density array. 
    freqBandwidth: scalar, optional
        Frequency bandwidth used to generate the time series from psd.
        Default to None, every frequency in freq will be used. 
    
    Returns
    -------
    ts: 1darray
        Array containing all the time data for the time series.
    amps: 1darray
        Amplitude array containing the amplitudes of the time series 
        corresponding to ts.

    Raises
    ------
    ValueError
        If the fs or time is not a scalar.
        If freq or psd is not a 1darray or has less than 3 elements.
        If freq and psd are in different lengths.
        If freq contains negative elements.
        If freq is not equally spaced increasing. 

    Examples
    --------
    >>> from ffpack.lsg import harmonicSuperposition
    >>> fs = 100
    >>> time = 10
    >>> freq = [ 0, 0.1, 0.2, 0.3, 0.4, 0.5 ]
    >>> psd = [ 0.01, 2, 0.05, 0.04, 0.01, 0.03 ]
    >>> ts, amps = harmonicSuperposition( fs, time, freq, psd, freqBandwidth=None )
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
    
    # edge case check for freq
    freq = np.array( freq, dtype=float )
    if len( freq.shape ) != 1:
        raise ValueError( "freq dimension should be 1" )
    if freq.shape[ 0 ] < 3:
        raise ValueError( "freq length should be at least 3" )
    if freq[ 0 ] < 0 or freq[ 1 ] < 0:
        raise ValueError( "freq array should be non-negative" )
    if freq[ 1 ] <= freq[ 0 ]:
        raise ValueError( "freq array should be strictly increasing" )
    diff = freq[ 1 ] - freq[ 0 ]
    tolerance = np.power( 10.0, -4 )
    for i in range( 1, len( freq ) ):
        if abs( freq[ i ] - freq[ i - 1 ] - diff ) > tolerance:
            raise ValueError( "freq array should be equally spaced increasing" )
    
    # edge case check for psd
    psd = np.array( psd, dtype=float )
    if len( psd.shape ) != 1:
        raise ValueError( "psd dimension should be 1" )
    if psd.shape[ 0 ] < 3:
        raise ValueError( "psd length should be at least 3" )
    
    # edge case check for freq and psd length
    if len( freq ) != len( psd ):
        raise ValueError( "freq and psd should be in the same length" )

    # edge case check for freqBandwidth
    if freqBandwidth is not None:
        if not isinstance( freqBandwidth, int ) and not isinstance( freqBandwidth, float ):
            raise ValueError( "freqBandwidth should be a scalar" )
        if freqBandwidth <= 0:
            raise ValueError( "freqBandwidth should be positive" )

    # deal with freqBandwidth
    if freqBandwidth is None or freqBandwidth < freq[ 1 ] - freq[ 0 ]:
        freqBandwidth = freq[ 1 ] - freq[ 0 ]
    next = round( freqBandwidth / ( freq[ 1 ] - freq[ 0 ] ) )

    n = round( fs * time )
    ts = 1 / fs * np.arange( n, dtype=float )
    amps = np.zeros( n )
    # generate phase angle
    phis = -np.pi + 2 * np.pi * np.random.randn( len( freq ) )

    i = 0
    while i < len( freq ):
        for j in range( n ):
            amps[ j ] += np.sqrt( 2 * psd[ i ] * freqBandwidth ) * \
                np.sin( 2 * np.pi * freq[ i ] * ts[ j ] + phis[ i ] )
        i += next
    return ts, amps
