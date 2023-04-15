#!/usr/bin/env python3

import numpy as np
from scipy import signal


def periodogramSpectrum( data, fs ):
    '''
    Power spectral density with `scipy.signal.periodogram`.

    Parameters
    ----------
    data: 1darray
        Sequence to calculate power spectral density.
    fs: scalar
        Sampling frequency.
    
    Returns
    -------
    freq: 1darray
        frequency components.
    psd: 1darray
        Power spectral density. 
    
    Raises
    ------
    ValueError
        If data is not a 1darray.
        If fs is not a scalar.

    Examples
    --------
    >>> from ffpack.lsm import periodogramSpectrum
    >>> data = [ 2, 5, 3, 6, 2, 4, 1, 6, 1, 3, 1, 5, 3, 6, 3, 6, 4, 5, 2 ]
    >>> fs = 2
    >>> freq, psd = periodogramSpectrum( data, fs )
    '''
    data = np.array( data, dtype=float )
    if len( data.shape ) != 1:
        raise ValueError( "Input data dimension should be 1" )
    if data.shape[ 0 ] < 2:
        raise ValueError( "Input data length should be at least 2" )
    if not isinstance( fs, int ) and not isinstance( fs, float ):
        raise ValueError( "fs should be a scalar" )

    ( freq, psd ) = signal.periodogram( data, fs, scaling='density' )
    return freq, psd


def welchSpectrum( data, fs, nperseg=1024 ):
    '''
    Power spectral density with `scipy.signal.welch`.

    Parameters
    ----------
    data: 1darray
        Sequence to calculate power spectral density.
    fs: scalar
        Sampling frequency.
    nperseg: scalar
        Length of each segment. Defaults to 1024.
    
    Returns
    -------
    freq: 1darray
        frequency components.
    psd: 1darray
        Power spectral density. 
    
    Raises
    ------
    ValueError
        If data is not a 1darray.
        If fs is not a scalar.

    Examples
    --------
    >>> from ffpack.lsm import welchSpectrum
    >>> data = [ 2, 5, 3, 6, 2, 4, 1, 6, 1, 3, 1, 5, 3, 6, 3, 6, 4, 5, 2 ]
    >>> fs = 2
    >>> freq, psd = welchSpectrum( data, fs, nperseg=1024 )
    '''
    data = np.array( data, dtype=float )
    if len( data.shape ) != 1:
        raise ValueError( "Input data dimension should be 1" )
    if data.shape[ 0 ] < 2:
        raise ValueError( "Input data length should be at least 2" )
    if not isinstance( fs, int ) and not isinstance( fs, float ):
        raise ValueError( "fs should be a scalar" )

    ( freq, psd ) = signal.welch( data, fs, nperseg=nperseg )
    return freq, psd
