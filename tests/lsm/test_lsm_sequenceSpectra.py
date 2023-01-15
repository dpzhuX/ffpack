#!/usr/bin/env python3

from ffpack import lsm
import numpy as np
import pytest


###############################################################################
# Test periodogramSpectrum
###############################################################################
def test_periodogramSpectrum_incorrectDataCase_valueError():
    fs = 10

    # case 1: data is a scalar
    data = 10
    with pytest.raises( ValueError ):
        _ = lsm.periodogramSpectrum( data, fs )

    # case 2: data is empty
    data = [ ]
    with pytest.raises( ValueError ):
        _ = lsm.periodogramSpectrum( data, fs )

    # case 3: data has only one point
    data = [ 1.0 ]
    with pytest.raises( ValueError ):
        _ = lsm.periodogramSpectrum( data, fs )
    
    # case 4: data is two dimensional
    data = [ [ ] ]
    with pytest.raises( ValueError ):
        _ = lsm.periodogramSpectrum( data, fs )


def test_periodogramSpectrum_incorrectFsCase_valueError():
    data = [ 2, 5, 3, 6, 2, 4, 1, 6, 1, 3, 1, 5, 3, 6, 3, 6, 4, 5, 2 ]

    # case 1: fs is not a scalar
    fs = [ ]
    with pytest.raises( ValueError ):
        _ = lsm.periodogramSpectrum( data, fs )

    # case 2: fs is two dimensional
    fs = [ [ ] ]
    with pytest.raises( ValueError ):
        _ = lsm.periodogramSpectrum( data, fs )


def test_periodogramSpectrum_onePeak_samePeaks():
    np.random.seed( 0 )
    gfs = 1000
    fs1 = 10
    T = 10
    n0 = -10

    t = np.r_[ 0: T: ( 1 / gfs ) ] 
    gdata = np.sin( 2 * fs1 * np.pi * t )
    gdata += np.random.randn( len( gdata ) ) * 10**( n0 / 20.0 ) 
    gfreq, gpsd = lsm.periodogramSpectrum( gdata, gfs )

    ind = np.argpartition( gpsd, -1 )[ -1: ]
    np.testing.assert_allclose( gfreq[ ind[ 0 ] ], fs1 )


def test_periodogramSpectrum_twoPeaks_samePeaks():
    np.random.seed( 0 )
    gfs = 1000
    fs1 = 10
    fs2 = 60
    T = 10
    n0 = -10

    t = np.r_[ 0: T: ( 1 / gfs ) ] 
    gdata = np.sin( 2 * fs1 * np.pi * t ) + np.sin( 2 * fs2 * np.pi * t ) 
    gdata += np.random.randn( len( gdata ) ) * 10**( n0 / 20.0 ) 
    gfreq, gpsd = lsm.periodogramSpectrum( gdata, gfs )

    ind = np.argpartition( gpsd, -2 )[ -2: ]
    peak1 = min( gfreq[ ind ] )
    peak2 = max( gfreq[ ind ] )
    np.testing.assert_allclose( [ peak1, peak2 ], [ fs1, fs2 ] )


def test_periodogramSpectrum_threePeaks_samePeaks():
    np.random.seed( 0 )
    gfs = 1000
    fs1 = 10
    fs2 = 60
    fs3 = 100
    T = 10
    n0 = -10

    t = np.r_[ 0: T: ( 1 / gfs ) ] 
    gdata = np.sin( 2 * fs1 * np.pi * t ) + np.sin( 2 * fs2 * np.pi * t ) + \
        np.sin( 2 * fs3 * np.pi * t )
    gdata += np.random.randn( len( gdata ) ) * 10**( n0 / 20.0 ) 
    gfreq, gpsd = lsm.periodogramSpectrum( gdata, gfs )

    ind = np.argpartition( gpsd, -3 )[ -3: ]
    peak1 = min( gfreq[ ind ] )
    peak3 = max( gfreq[ ind ] )
    peak2 = sum( gfreq[ ind ] ) - peak1 - peak3
    np.testing.assert_allclose( [ peak1, peak2, peak3 ], [ fs1, fs2, fs3 ] )
