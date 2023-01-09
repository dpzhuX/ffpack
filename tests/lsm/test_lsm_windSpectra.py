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
    x = 1200 * n / delta1
    expectedRst = 4.0 * x * x / np.power( 1 + x * x, 4 / 3 )
    np.testing.assert_allclose( np.round( calRst, 4 ), np.round( expectedRst, 4 ) )

    # case 2: not normalized 
    calRst = lsm.davenportSpectrumWithDragCoef( n, delta1, normalized=False )
    expectedRst = expectedRst * kappa * delta1 * delta1 / n
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
    uf = 0.4 * uz / np.log( z / z0 )
    expectedRst = expectedRst * uf * uf / n
    np.testing.assert_allclose( np.round( calRst, 4 ), np.round( expectedRst, 4 ) )


def test_davenportSpectrum_normalized_sameRst():
    # The normalized power spectrum density should be the same 
    # with the two Davenport spectrum equations when the wind speed is the same
    n = 2
    delta1 = uz = 10

    calDrag = lsm.davenportSpectrumWithDragCoef( n, delta1 )
    calRoughness = lsm.davenportSpectrumWithRoughnessLength( n, uz )
    np.testing.assert_allclose( calDrag, calRoughness )
