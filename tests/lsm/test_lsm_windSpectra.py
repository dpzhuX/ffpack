#!/usr/bin/env python3

from ffpack import lsm
import numpy as np
import pytest


###############################################################################
# Test davenportSpectrumWithDragCoef
###############################################################################
def test_davenportSpectrumWithDragCoef_inputNotScalarCase_valueError():
    n = 2
    delta1 = 10

    # case 1: n is not a scalar
    with pytest.raises( ValueError ):
        _ = lsm.davenportSpectrumWithDragCoef( [ ], delta1 )

    # case 2: delata1 is not a scalar
    with pytest.raises( ValueError ):
        _ = lsm.davenportSpectrumWithDragCoef( n, [ ] )


def test_davenportSpectrumWithDragCoef_normalUseCase_expectedRst():
    kappa = 0.005
    n = 2
    delta1 = 10

    # case 1: normalized
    calRst = lsm.davenportSpectrumWithDragCoef( n, delta1 )
    x = 120 * n
    expectedRst = 4.0 * x * x / np.power( 1 + x * x, 4 / 3 )
    np.testing.assert_allclose( np.round( calRst, 4 ), np.round( expectedRst, 4 ) )

    # case 2: not normalized 
    calRst = lsm.davenportSpectrumWithDragCoef( n, delta1, normalized=False )
    x = 1200 * n / delta1
    right = 4.0 * x * x / np.power( 1 + x * x, 4 / 3 )
    expectedRst = right * kappa * delta1 * delta1 / n
    np.testing.assert_allclose( np.round( calRst, 4 ), np.round( expectedRst, 4 ) )



###############################################################################
# Test davenportSpectrumWithRoughnessLength
###############################################################################
def test_davenportSpectrumWithRoughnessLength_inputNotScalarCase_valueError():
    n = 2
    uz = 10

    # case 1: n is not a scalar
    with pytest.raises( ValueError ):
        _ = lsm.davenportSpectrumWithRoughnessLength( [ ], uz )

    # case 2: uz is not a scalar
    with pytest.raises( ValueError ):
        _ = lsm.davenportSpectrumWithRoughnessLength( n, [ ] )


def test_davenportSpectrumWithRoughnessLength_normalUseCase_expectedRst():
    z = 10
    z0 = 0.03
    n = 2
    uz = 10

    # case 1: normalized
    calRst = lsm.davenportSpectrumWithRoughnessLength( n, uz )
    x = 1200 * n / uz
    expectedRst = 4.0 * x * x / np.power( 1 + x * x, 4 / 3 )
    np.testing.assert_allclose( np.round( calRst, 4 ), np.round( expectedRst, 4 ) )

    # case 2: not normalized 
    calRst = lsm.davenportSpectrumWithRoughnessLength( n, uz, normalized=False )
    x = 1200 * n / uz
    right = 4.0 * x * x / np.power( 1 + x * x, 4 / 3 )
    uf = 0.4 * uz / np.log( z / z0 )
    expectedRst = right * uf * uf / n
    np.testing.assert_allclose( np.round( calRst, 4 ), np.round( expectedRst, 4 ) )


def test_davenportSpectrum_normalized_sameRst():
    # The normalized power spectrum density should be the same 
    # with the two Davenport spectrum equations when the wind speed is the same
    n = 2
    delta1 = uz = 10

    calDrag = lsm.davenportSpectrumWithDragCoef( n, delta1 )
    calRoughness = lsm.davenportSpectrumWithRoughnessLength( n, uz )
    np.testing.assert_allclose( calDrag, calRoughness )



###############################################################################
# Test ec1Spectrum
###############################################################################
def test_ec1Spectrum_inputNotScalarCase_valueError():
    n = 2
    uz = 10

    # case 1: n is not a scalar
    with pytest.raises( ValueError ):
        _ = lsm.ec1Spectrum( [ ], uz )

    # case 2: uz is not a scalar
    with pytest.raises( ValueError ):
        _ = lsm.ec1Spectrum( n, [ ] )


def test_ec1Spectrum_tcatIncorrect_valueError():
    n = 2
    uz = 10

    # case 1: tcat is not a scalar
    with pytest.raises( ValueError ):
        _ = lsm.ec1Spectrum( n, uz, tcat=[ ] )

    # case 2: tcat < 0
    with pytest.raises( ValueError ):
        _ = lsm.ec1Spectrum( n, uz, tcat=-1 )

    # case 3: tcat > 4
    with pytest.raises( ValueError ):
        _ = lsm.ec1Spectrum( n, uz, tcat=5 )


def test_ec1Spectrum_normalizedUseCase_expectedRst():
    n = 2
    uz = 10
    calRst = lsm.ec1Spectrum( n, uz )
    expectedRst = 6.8 * n / np.power( 1 + 10.2 * n, 5 / 3 )
    np.testing.assert_allclose( np.round( calRst, 4 ), np.round( expectedRst, 4 ) )


def test_ec1Spectrum_notNormalizedUseCase_expectedRst():
    n = 2
    uz = 10
    sigma = 5

    def lzCalc( z0, zmin, z ):
        alpha = 0.67 + 0.05 * np.log( z0 )
        if z < zmin:
            rst = 300 * np.power( zmin / 200, alpha )
            return rst
        
        rst = 300 * np.power( z / 200, alpha )
        return rst
    
    def rstCalc( lz ):
        fl = n * lz / uz
        right = 6.8 * fl / np.power( 1 + 10.2 * fl, 5 / 3 )
        rst = right * sigma * sigma / n
        return rst

    # case 1: tcat = 0 
    tcat = 0
    # z > zmin
    z = 20
    calRst = lsm.ec1Spectrum( n, uz, sigma=sigma, z=z, tcat=tcat, normalized=False )
    expectedRst = rstCalc( lzCalc( 0.003, 1, z ) )
    np.testing.assert_allclose( np.round( calRst, 4 ), np.round( expectedRst, 4 ) )
    # z < zmin
    z = 0.8
    calRst = lsm.ec1Spectrum( n, uz, sigma=sigma, z=z, tcat=tcat, normalized=False )
    expectedRst = rstCalc( lzCalc( 0.003, 1, z ) )
    np.testing.assert_allclose( np.round( calRst, 4 ), np.round( expectedRst, 4 ) )

    # case 2: tcat = 1
    tcat = 1
    # z > zmin
    z = 20
    calRst = lsm.ec1Spectrum( n, uz, sigma=sigma, z=z, tcat=tcat, normalized=False )
    expectedRst = rstCalc( lzCalc( 0.01, 1, z ) )
    np.testing.assert_allclose( np.round( calRst, 4 ), np.round( expectedRst, 4 ) )
    # z < zmin
    z = 0.8
    calRst = lsm.ec1Spectrum( n, uz, sigma=sigma, z=z, tcat=tcat, normalized=False )
    expectedRst = rstCalc( lzCalc( 0.01, 1, z ) )
    np.testing.assert_allclose( np.round( calRst, 4 ), np.round( expectedRst, 4 ) )

    # case 3: tcat = 2
    tcat = 2
    # z > zmin
    z = 20
    calRst = lsm.ec1Spectrum( n, uz, sigma=sigma, z=z, tcat=tcat, normalized=False )
    expectedRst = rstCalc( lzCalc( 0.05, 2, z ) )
    np.testing.assert_allclose( np.round( calRst, 4 ), np.round( expectedRst, 4 ) )
    # z < zmin
    z = 0.8
    calRst = lsm.ec1Spectrum( n, uz, sigma=sigma, z=z, tcat=tcat, normalized=False )
    expectedRst = rstCalc( lzCalc( 0.05, 2, z ) )
    np.testing.assert_allclose( np.round( calRst, 4 ), np.round( expectedRst, 4 ) )

    # case 4: tcat = 3
    tcat = 3
    # z > zmin
    z = 20
    calRst = lsm.ec1Spectrum( n, uz, sigma=sigma, z=z, tcat=tcat, normalized=False )
    expectedRst = rstCalc( lzCalc( 0.3, 5, z ) )
    np.testing.assert_allclose( np.round( calRst, 4 ), np.round( expectedRst, 4 ) )
    # z < zmin
    z = 0.8
    calRst = lsm.ec1Spectrum( n, uz, sigma=sigma, z=z, tcat=tcat, normalized=False )
    expectedRst = rstCalc( lzCalc( 0.3, 5, z ) )
    np.testing.assert_allclose( np.round( calRst, 4 ), np.round( expectedRst, 4 ) )

    # case 5: tcat = 4
    tcat = 4
    # z > zmin
    z = 20
    calRst = lsm.ec1Spectrum( n, uz, sigma=sigma, z=z, tcat=tcat, normalized=False )
    expectedRst = rstCalc( lzCalc( 1.0, 10, z ) )
    np.testing.assert_allclose( np.round( calRst, 4 ), np.round( expectedRst, 4 ) )
    # z < zmin
    z = 0.8
    calRst = lsm.ec1Spectrum( n, uz, sigma=sigma, z=z, tcat=tcat, normalized=False )
    expectedRst = rstCalc( lzCalc( 1.0, 10, z ) )
    np.testing.assert_allclose( np.round( calRst, 4 ), np.round( expectedRst, 4 ) )



###############################################################################
# Test iecSpectrum
###############################################################################
def test_iecSpectrum_inputNotScalarCase_valueError():
    f = 2
    vhub = 10

    # case 1: n is not a scalar
    with pytest.raises( ValueError ):
        _ = lsm.iecSpectrum( [ ], vhub )

    # case 2: vhub is not a scalar
    with pytest.raises( ValueError ):
        _ = lsm.iecSpectrum( f, [ ] )


def test_iecSpectrum_kIncorrect_valueError():
    f = 2
    vhub = 10

    # case 1: k is not a scalar
    with pytest.raises( ValueError ):
        _ = lsm.iecSpectrum( f, vhub, k=[ ] )

    # case 2: k < 1
    with pytest.raises( ValueError ):
        _ = lsm.iecSpectrum( f, vhub, k=0 )

    # case 3: k > 3
    with pytest.raises( ValueError ):
        _ = lsm.iecSpectrum( f, vhub, k=4 )


def test_iecSpectrum_normalizedUseCase_expectedRst():
    f = 2
    vhub = 100
    calRst = lsm.iecSpectrum( f, vhub )
    expectedRst = 4 * f / np.power( 1 + 6 * f, 5 / 3 )
    np.testing.assert_allclose( np.round( calRst, 4 ), np.round( expectedRst, 4 ) )


def test_iecSpectrum_notNormalizedUseCase_expectedRst():
    f = 2
    vhub = 10
    sigma = 5
    
    def rstCalc( sigmak, lk ):
        nf = f * lk / vhub
        right = 4 * nf / np.power( 1 + 6 * nf, 5 / 3 )
        rst = right * sigmak * sigmak / f
        return rst

    # case 1: k = 1
    k = 1
    # z > 60
    z = 80
    calRst = lsm.iecSpectrum( f, vhub, sigma=sigma, z=z, k=k, normalized=False )
    expectedRst = rstCalc( sigma, 42 * 8.1 )
    np.testing.assert_allclose( np.round( calRst, 4 ), np.round( expectedRst, 4 ) )
    # z < 60
    z = 40
    calRst = lsm.iecSpectrum( f, vhub, sigma=sigma, z=z, k=k, normalized=False )
    expectedRst = rstCalc( sigma, 0.7 * z * 8.1 )
    np.testing.assert_allclose( np.round( calRst, 4 ), np.round( expectedRst, 4 ) )

    # case 2: k = 2
    k = 2
    # z > 60
    z = 80
    calRst = lsm.iecSpectrum( f, vhub, sigma=sigma, z=z, k=k, normalized=False )
    expectedRst = rstCalc( sigma * 0.8, 42 * 2.7 )
    np.testing.assert_allclose( np.round( calRst, 4 ), np.round( expectedRst, 4 ) )
    # z < 60
    z = 40
    calRst = lsm.iecSpectrum( f, vhub, sigma=sigma, z=z, k=k, normalized=False )
    expectedRst = rstCalc( sigma * 0.8, 0.7 * z * 2.7 )
    np.testing.assert_allclose( np.round( calRst, 4 ), np.round( expectedRst, 4 ) )

    # case 3: k = 3
    k = 3
    # z > 60
    z = 80
    calRst = lsm.iecSpectrum( f, vhub, sigma=sigma, z=z, k=k, normalized=False )
    expectedRst = rstCalc( sigma * 0.5, 42 * 0.66 )
    np.testing.assert_allclose( np.round( calRst, 4 ), np.round( expectedRst, 4 ) )
    # z < 60
    z = 40
    calRst = lsm.iecSpectrum( f, vhub, sigma=sigma, z=z, k=k, normalized=False )
    expectedRst = rstCalc( sigma * 0.5, 0.7 * z * 0.66 )
    np.testing.assert_allclose( np.round( calRst, 4 ), np.round( expectedRst, 4 ) )



###############################################################################
# Test apiSpectrum
###############################################################################
def test_apiSpectrum_inputNotScalarCase_valueError():
    f = 2
    u0 = 10

    # case 1: f is not a scalar
    with pytest.raises( ValueError ):
        _ = lsm.apiSpectrum( [ ], u0 )

    # case 2: u0 is not a scalar
    with pytest.raises( ValueError ):
        _ = lsm.apiSpectrum( f, [ ] )


def test_apiSpectrum_normalUseCase_expectedRst():
    f = 2
    u0 = 10
    calRst = lsm.apiSpectrum( f, u0, z=10 )

    n = 0.468
    ftilde = 172 * f * np.power( u0 / 10, -0.75 )
    expectedRst = 320 * np.power( u0 / 10, 2 ) 
    expectedRst = expectedRst / np.power( 1 + np.power( ftilde, n ), 5 / ( 3 * n ) )
    np.testing.assert_allclose( np.round( calRst, 4 ), np.round( expectedRst, 4 ) )
