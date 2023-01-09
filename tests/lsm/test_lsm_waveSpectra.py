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



###############################################################################
# Test isscSpectrum
###############################################################################
def test_isscSpectrum_inputNotScalarCase_valueError():
    # case 1: w is not a scalar
    w = [ ]
    wp = 0.2
    Hs = 20
    with pytest.raises( ValueError ):
        _ = lsm.isscSpectrum( w, wp, Hs )

    # case 2: wp is not a scalar
    w = 0.1
    wp = [ ]
    Hs = 20
    with pytest.raises( ValueError ):
        _ = lsm.isscSpectrum( w, wp, Hs )

    # case 3: Hs is not a scalar
    w = 0.1
    wp = 0.2
    Hs = [ ]
    with pytest.raises( ValueError ):
        _ = lsm.isscSpectrum( w, wp, Hs )
    

def test_isscSpectrum_normalUseCase_expectedRst():
    wp = 0.04
    Hs = 20

    # case 1: w = wp
    w = wp
    calRst = lsm.isscSpectrum( w, wp, Hs )
    expectedRst = 5 / 16 * Hs * Hs / w * np.exp( -1.25 )
    np.testing.assert_allclose( np.round( calRst, 4 ), np.round( expectedRst, 4 ) )

    # case 2: w = 2 * wp
    w = wp * 2
    calRst = lsm.isscSpectrum( w, wp, Hs )
    expectedRst = 5 / 16 * Hs * Hs / 16 / w * np.exp( -1.25 / 16 )
    np.testing.assert_allclose( np.round( calRst, 4 ), np.round( expectedRst, 4 ) )

    # case 3: w = wp / 2
    w = wp / 2
    calRst = lsm.isscSpectrum( w, wp, Hs )
    expectedRst = 5 / 16 * Hs * Hs * 16 / w * np.exp( -1.25 * 16 )
    np.testing.assert_allclose( np.round( calRst, 6 ), np.round( expectedRst, 6 ) )



###############################################################################
# Test gaussianSwellSpectrum
###############################################################################
def test_gaussianSwellSpectrum_inputNotScalarCase_valueError():
    # case 1: w is not a scalar
    w = [ ]
    wp = 0.2
    Hs = 20
    sigma = 0.07
    with pytest.raises( ValueError ):
        _ = lsm.gaussianSwellSpectrum( w, wp, Hs, sigma )

    # case 2: wp is not a scalar
    w = 0.1
    wp = [ ]
    Hs = 20
    sigma = 0.07
    with pytest.raises( ValueError ):
        _ = lsm.gaussianSwellSpectrum( w, wp, Hs, sigma )

    # case 3: Hs is not a scalar
    w = 0.1
    wp = 0.2
    Hs = [ ]
    sigma = 0.07
    with pytest.raises( ValueError ):
        _ = lsm.gaussianSwellSpectrum( w, wp, Hs, sigma )

    # case 4: sigma is not a scalar
    w = 0.1
    wp = 0.2
    Hs = 20
    sigma = [ ]
    with pytest.raises( ValueError ):
        _ = lsm.gaussianSwellSpectrum( w, wp, Hs, sigma )


def test_gaussianSwellSpectrum_normalUseCase_expectedRst():
    wp = 0.04
    Hs = 20
    sigma = 0.07

    # case 1: w = wp
    w = wp
    calRst = lsm.gaussianSwellSpectrum( w, wp, Hs, sigma )
    expectedRst = Hs * Hs / ( 16 * sigma * np.power( 2 * np.pi, 1.5 ) )
    np.testing.assert_allclose( np.round( calRst, 4 ), np.round( expectedRst, 4 ) )

    # case 2: w = 2 * wp
    w = wp * 2
    calRst = lsm.gaussianSwellSpectrum( w, wp, Hs, sigma )
    pexp = np.power( wp / ( 2 * np.pi * sigma ), 2 ) / 2
    expectedRst = Hs * Hs / ( 16 * sigma * np.power( 2 * np.pi, 1.5 ) ) * np.exp( -pexp )
    np.testing.assert_allclose( np.round( calRst, 4 ), np.round( expectedRst, 4 ) )

    # case 3: w = wp / 2
    w = wp / 2
    calRst = lsm.gaussianSwellSpectrum( w, wp, Hs, sigma )
    pexp = np.power( wp / ( 4 * np.pi * sigma), 2 ) / 2
    expectedRst = Hs * Hs / ( 16 * sigma * np.power( 2 * np.pi, 1.5 ) ) * np.exp( -pexp )
    np.testing.assert_allclose( np.round( calRst, 4 ), np.round( expectedRst, 4 ) )



###############################################################################
# Test ochiHubbleSpectrum
###############################################################################
def test_ochiHubbleSpectrum_inputNotScalarCase_valueError():
    w = 0.1
    wp1 = 0.2
    wp2 = 0.3
    Hs1 = 20
    Hs2 = 15
    lambda1 = 1.5
    lambda2 = 2.5

    # case 1: w is not a scalar
    with pytest.raises( ValueError ):
        _ = lsm.ochiHubbleSpectrum( [ ], wp1, wp2, Hs1, Hs2, lambda1, lambda2 )

    # case 2: wp1 or wp2 is not a scalar
    with pytest.raises( ValueError ):
        _ = lsm.ochiHubbleSpectrum( w, [ ], wp2, Hs1, Hs2, lambda1, lambda2 )
    with pytest.raises( ValueError ):
        _ = lsm.ochiHubbleSpectrum( w, wp1, [ ], Hs1, Hs2, lambda1, lambda2 )

    # case 3: Hs1 or Hs2 is not a scalar
    with pytest.raises( ValueError ):
        _ = lsm.ochiHubbleSpectrum( w, wp1, wp2, [ ], Hs2, lambda1, lambda2 )
    with pytest.raises( ValueError ):
        _ = lsm.ochiHubbleSpectrum( w, wp1, wp2, Hs1, [ ], lambda1, lambda2 )

    # case 4: lambda1 or lambda2 is not a scalar
    with pytest.raises( ValueError ):
        _ = lsm.ochiHubbleSpectrum( w, wp1, wp2, Hs1, Hs2, [ ], lambda2 )
    with pytest.raises( ValueError ):
        _ = lsm.ochiHubbleSpectrum( w, wp1, wp2, Hs1, Hs2, lambda1, [ ] )


def test_ochiHubbleSpectrum_wp1NotSmallerThanwp2_valueError():
    w = 0.1
    wp2 = 0.3
    Hs1 = 20
    Hs2 = 15
    lambda1 = 1.5
    lambda2 = 2.5

    # case 1: wp1 = wp2
    wp1 = wp2
    with pytest.raises( ValueError ):
        _ = lsm.ochiHubbleSpectrum( w, wp1, wp2, Hs1, Hs2, lambda1, lambda2 )
    
    # case 2: wp1 > wp2
    wp1 = wp2 * 2
    with pytest.raises( ValueError ):
        _ = lsm.ochiHubbleSpectrum( w, wp1, wp2, Hs1, Hs2, lambda1, lambda2 )


def test_ochiHubbleSpectrum_normalUseCase_expectedRst():
    w = 0.1
    wp1 = 0.2
    wp2 = 0.3
    Hs1 = 20000
    Hs2 = 15000
    lambda1 = 1.5
    lambda2 = 2.5

    calRst = lsm.ochiHubbleSpectrum( w, wp1, wp2, Hs1, Hs2, lambda1, lambda2 )
    expectedRst = 0.1156
    np.testing.assert_allclose( np.round( calRst, 4 ), np.round( expectedRst, 4 ) )
