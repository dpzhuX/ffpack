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
    freq = [ 1, 2, 3, 4, 5 ]
    psd = [ 1, 2, 5, 2, 1 ]

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


def test_harmonicSuperposition_freqOrPsdIncorrect_valueError():
    fs = 100
    time = 10
    freq = [ 1, 2, 3, 4, 5 ]
    psd = [ 1, 2, 5, 2, 1, 3 ]

    # case 1: freq is incorrect
    with pytest.raises( ValueError ):
        _ = lsg.harmonicSuperposition( fs, time, 1.0, psd )
    with pytest.raises( ValueError ):
        _ = lsg.harmonicSuperposition( fs, time, [ [ ] ], psd )

    # case 2: psd is incorrect
    with pytest.raises( ValueError ):
        _ = lsg.harmonicSuperposition( fs, time, freq, 1.0 )
    with pytest.raises( ValueError ):
        _ = lsg.harmonicSuperposition( fs, time, freq, [ [ ] ] )

    # case 3: freq and psd are in different lengths
    with pytest.raises( ValueError ):
        _ = lsg.harmonicSuperposition( fs, time, freq, psd )

    # case 4: freq contains negative elements
    freq = [ -1, 2, 3, 4, 5 ]
    with pytest.raises( ValueError ):
        _ = lsg.harmonicSuperposition( fs, time, freq, psd )
    
    # case 5: freq is not strictly increasing
    freq = [ 1, 2, 5, 4, 5 ]
    with pytest.raises( ValueError ):
        _ = lsg.harmonicSuperposition( fs, time, freq, psd )
    

def test_randomWalkUniform_normalUseCase_diffByOne():
    fs = 100
    time = 10
    freq = [ 0, 0.1, 0.2, 0.3, 0.4, 0.5 ]
    psd = [ 0.01, 2, 0.05, 0.04, 0.01, 0.03 ]

    ts, rst = lsg.harmonicSuperposition( fs, time, freq, psd )
