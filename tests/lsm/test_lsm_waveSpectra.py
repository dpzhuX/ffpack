#!/usr/bin/env python3

from ffpack import lsm
import numpy as np
import pytest


###############################################################################
# Test piersonMoskowitzSpectrum
###############################################################################
def test_piersonMoskowitzSpectrum_inputNotScalarCase_valueError():
    # case 1: w is not a scalar
    w = [ ]
    wp = 0.2
    with pytest.raises( ValueError ):
        _ = lsm.piersonMoskowitzSpectrum( w, wp )

    # case 2: wp is not a scalar
    w = 0.1
    wp = [ ]
    with pytest.raises( ValueError ):
        _ = lsm.piersonMoskowitzSpectrum( w, wp )


def test_piersonMoskowitzSpectrum_normalUseCase_expectedRst():
    alpha = 0.0081
    beta = 0.74
    g = 9.81

    Uw = 20

    # case 1: w = wp
    w = g / Uw
    calRst = lsm.piersonMoskowitzSpectrum( w, Uw, alpha, beta, g )
    expectedRst = alpha * g * g / np.power( w, 5 ) * np.exp( -beta ) 
    np.testing.assert_allclose( np.round( calRst, 4 ), np.round( expectedRst, 4 ) )

    # case 2: w < wp
    w = g / Uw / 2
    calRst = lsm.piersonMoskowitzSpectrum( w, Uw, alpha, beta, g )
    expectedRst = alpha * g * g / np.power( w, 5 ) * np.exp( -beta * 16 ) 
    np.testing.assert_allclose( np.round( calRst, 4 ), np.round( expectedRst, 4 ) )

    # case 3: w > wp
    w = g / Uw * 2
    calRst = lsm.piersonMoskowitzSpectrum( w, Uw, alpha, beta, g )
    expectedRst = alpha * g * g / np.power( w, 5 ) * np.exp( -beta / 16 ) 
    np.testing.assert_allclose( np.round( calRst, 4 ), np.round( expectedRst, 4 ) )



###############################################################################
# Test jonswapSpectrum
###############################################################################
def test_jonswapSpectrum_inputNotScalarCase_valueError():
    # case 1: w is not a scalar
    w = [ ]
    wp = 0.2
    with pytest.raises( ValueError ):
        _ = lsm.jonswapSpectrum( w, wp )

    # case 2: wp is not a scalar
    w = 0.1
    wp = [ ]
    with pytest.raises( ValueError ):
        _ = lsm.jonswapSpectrum( w, wp )


def test_jonswapSpectrum_normalUseCase_expectedRst():
    alpha = 0.0081
    beta = 1.25
    gamma = 3.3
    g = 9.81

    wp = 0.04

    # case 1: w = wp
    w = wp
    calRst = lsm.jonswapSpectrum( w, wp, alpha, beta, gamma, g )
    expectedRst = alpha * g * g / np.power( w, 5 ) * np.exp( -beta ) * gamma
    np.testing.assert_allclose( np.round( calRst, 4 ), np.round( expectedRst, 4 ) )

    # case 2: w < wp
    w = wp / 2
    calRst = lsm.jonswapSpectrum( w, wp, alpha, beta, gamma, g )
    expectedRst = alpha * g * g / np.power( w, 5 ) * np.exp( -beta * 16 ) \
        * np.power( gamma, np.exp( -1 / 8.0 / ( 0.07 * 0.07 ) ) ) 
    np.testing.assert_allclose( np.round( calRst, 4 ), np.round( expectedRst, 4 ) )

    # case 3: w > wp
    w = wp * 2
    calRst = lsm.jonswapSpectrum( w, wp, alpha, beta, gamma, g )
    expectedRst = alpha * g * g / np.power( w, 5 ) * np.exp( -beta / 16 ) \
        * np.power( gamma, np.exp( -1 / 2.0 / ( 0.09 * 0.09 ) ) )
    np.testing.assert_allclose( np.round( calRst, 4 ), np.round( expectedRst, 4 ) )
