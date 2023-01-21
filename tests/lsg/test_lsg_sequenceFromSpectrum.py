#!/usr/bin/env python3

from ffpack import lsg
import numpy as np
import pytest


###############################################################################
# Test harmonicSuperposition
###############################################################################
def test_harmonicSuperposition_fsOrTimeIncorrect_valueError():
    fs = 100
    time = 10
    freq = [ 0, 0.1, 0.2, 0.3, 0.4, 0.5 ]
    psd = [ 0.01, 2, 0.05, 0.04, 0.01, 0.03 ]

    # case 1: fs is incorrect
    with pytest.raises( ValueError ):
        _ = lsg.harmonicSuperposition( [ ], time, freq, psd )
    with pytest.raises( ValueError ):
        _ = lsg.harmonicSuperposition( -1, time, freq, psd )
    
    # case 2: time is incorrect
    with pytest.raises( ValueError ):
        _ = lsg.harmonicSuperposition( fs, [ ], freq, psd )
    with pytest.raises( ValueError ):
        _ = lsg.harmonicSuperposition( fs, -1, freq, psd )


def test_harmonicSuperposition_freqIncorrect_valueError():
    fs = 100
    time = 10
    psd = [ 0.01, 2, 0.05, 0.04, 0.01, 0.03 ]

    # case 1: dimemsion is incorrect
    with pytest.raises( ValueError ):
        _ = lsg.harmonicSuperposition( fs, time, 1.0, psd )
    with pytest.raises( ValueError ):
        _ = lsg.harmonicSuperposition( fs, time, [ [ ] ], psd )

    # case 2: less than 3 elements
    with pytest.raises( ValueError ):
        _ = lsg.harmonicSuperposition( fs, time, [ 1.0, 2.0 ], psd )

    # case 3: freq contains negative elements
    freq = [ -0.1, 0.2, 0.3, 0.4, 0.5 ]
    with pytest.raises( ValueError ):
        _ = lsg.harmonicSuperposition( fs, time, freq, psd )
    freq = [ 0.1, -0.2, 0.3, 0.4, 0.5 ]
    with pytest.raises( ValueError ):
        _ = lsg.harmonicSuperposition( fs, time, freq, psd )
    
    # case 4: freq is not strictly increasing
    freq = [ 0.2, 0.1, 0.2, 0.3, 0.4, 0.5 ]
    with pytest.raises( ValueError ):
        _ = lsg.harmonicSuperposition( fs, time, freq, psd )
    freq = [ 0, 0.1, 0.2, 0.3, 0.5 ]
    with pytest.raises( ValueError ):
        _ = lsg.harmonicSuperposition( fs, time, freq, psd )


def test_harmonicSuperposition_psdIncorrect_valueError():    
    fs = 100
    time = 10
    freq = [ 0, 0.1, 0.2, 0.3, 0.4, 0.5 ]

    # case 1: dimemsion is incorrect
    with pytest.raises( ValueError ):
        _ = lsg.harmonicSuperposition( fs, time, freq, 1.0 )
    with pytest.raises( ValueError ):
        _ = lsg.harmonicSuperposition( fs, time, freq, [ [ ] ] )

    # case 2: less than 3 elements
    with pytest.raises( ValueError ):
        _ = lsg.harmonicSuperposition( fs, time, freq, [ 1.0, 2.0 ] )


def test_harmonicSuperposition_freqAndPsdDifferentLength_valueError(): 
    fs = 100
    time = 10
    freq = [ 0, 0.1, 0.2, 0.3, 0.4, 0.5 ]
    psd = [ 0.01, 2, 0.05 ]
    with pytest.raises( ValueError ):
        _ = lsg.harmonicSuperposition( fs, time, freq, psd )
    

def test_randomWalkUniform_normalUseCase_diffByOne():
    fs = 100
    time = 10
    freq = [ 0, 0.1, 0.2, 0.3, 0.4, 0.5 ]
    psd = [ 0.01, 2, 0.05, 0.04, 0.01, 0.03 ]

    # case 1: freqBandwidth not defined
    ts, rst = lsg.harmonicSuperposition( fs, time, freq, psd )
    np.testing.assert_allclose( len( ts ), fs * time )

    # case 2: user defined freqBandwidth
    ts, rst = lsg.harmonicSuperposition( fs, time, freq, psd, freqBandwidth=0.2 )
    np.testing.assert_allclose( len( ts ), fs * time )
