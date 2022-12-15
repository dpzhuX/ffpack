#!/usr/bin/env python3

from ffpack import utils
import numpy as np
import pytest

###############################################################################
# Test SnCurveFitter
###############################################################################
def test_snCurverFitter_oneDimData_valueError():
    data = [ 1.0, 2.5, 3.0, 4.5 ]
    with pytest.raises( ValueError ):
        _ = utils.SnCurveFitter( data, fatigueLimit=0.2 )


def test_snCurverFitter_emptyData_valueError():
    data = [ [ ] ]
    with pytest.raises( ValueError ):
        _ = utils.SnCurveFitter( data, fatigueLimit=0.2 )

    data = [ ]
    with pytest.raises( ValueError ):
        _ = utils.SnCurveFitter( data, fatigueLimit=0.2 )


def test_snCurverFitter_fatigueLimitLessZero_valueError():
    data = [ [ 10, 4 ], [ 10000, 1 ] ]
    with pytest.raises( ValueError ):
        _ = utils.SnCurveFitter( data, fatigueLimit=0 )

    with pytest.raises( ValueError ):
        _ = utils.SnCurveFitter( data, fatigueLimit=-1 )


def test_snCurverFitter_irregularData_valueError():
    data = [ [ 10, -4 ], [ -10000, 1 ] ]
    with pytest.raises( ValueError ):
        _ = utils.SnCurveFitter( data, fatigueLimit=0.5 )


def test_snCurverFitter_querySLessZero_valueError():
    data = [ [ 10, 4 ], [ 10000, 1 ] ]
    with pytest.raises( ValueError ):
        snCurveFitter = utils.SnCurveFitter( data, fatigueLimit=0.5 )
        _ = snCurveFitter.getN( 0 )

    with pytest.raises( ValueError ):
        snCurveFitter = utils.SnCurveFitter( data, fatigueLimit=0.5 )
        _ = snCurveFitter.getN( -1 )


def test_snCurverFitter_twoPairsData_queryPass():
    data = [ [ 10, 4 ], [ 10000, 1 ] ]
    snCurveFitter = utils.SnCurveFitter( data, fatigueLimit=0.5 )
    np.testing.assert_allclose( snCurveFitter.getN( 4 ), 1e1 )
    np.testing.assert_allclose( snCurveFitter.getN( 3 ), 1e2 )
    np.testing.assert_allclose( snCurveFitter.getN( 2 ), 1e3 )
    np.testing.assert_allclose( snCurveFitter.getN( 1 ), 1e4 )
    np.testing.assert_allclose( snCurveFitter.getN( 0.5 ), -1 )


def test_snCurverFitter_threePairsData_queryPass():
    data = [ [ 10, 5 ], [ 100, 4 ], [ 100000, 1 ] ]
    snCurveFitter = utils.SnCurveFitter( data, fatigueLimit=0.2 )
    np.testing.assert_allclose( snCurveFitter.getN( 5 ), 1e1 )
    np.testing.assert_allclose( snCurveFitter.getN( 4 ), 1e2 )
    np.testing.assert_allclose( snCurveFitter.getN( 3 ), 1e3 )
    np.testing.assert_allclose( snCurveFitter.getN( 2 ), 1e4 )
    np.testing.assert_allclose( snCurveFitter.getN( 1 ), 1e5 )
    np.testing.assert_allclose( snCurveFitter.getN( 0.1 ), -1 )


def test_snCurverFitter_fourPairsData_queryPass():
    data = [ [ 10, 5.5 ], [ 10000, 4 ], [ 1000000, 3 ], [ 10000000, 2.5 ] ]
    snCurveFitter = utils.SnCurveFitter( data, fatigueLimit=0.3 )
    np.testing.assert_allclose( snCurveFitter.getN( 5 ), 1e2 )
    np.testing.assert_allclose( snCurveFitter.getN( 4 ), 1e4 )
    np.testing.assert_allclose( snCurveFitter.getN( 3 ), 1e6 )
    np.testing.assert_allclose( snCurveFitter.getN( 2 ), 1e8 )
    np.testing.assert_allclose( snCurveFitter.getN( 1 ), 1e10 )
    np.testing.assert_allclose( snCurveFitter.getN( 0.3 ), -1 )
    np.testing.assert_allclose( snCurveFitter.getN( 0.2 ), -1 )
    np.testing.assert_allclose( snCurveFitter.getN( 0.1 ), -1 )


def test_snCurverFitter_fourPairsDataHighFatigueLimit_queryPass():
    data = [ [ 10, 5.5 ], [ 10000, 4 ], [ 1000000, 3 ], [ 10000000, 2.5 ] ]
    snCurveFitter = utils.SnCurveFitter( data, fatigueLimit=2.5 )
    np.testing.assert_allclose( snCurveFitter.getN( 5 ), 1e2 )
    np.testing.assert_allclose( snCurveFitter.getN( 4 ), 1e4 )
    np.testing.assert_allclose( snCurveFitter.getN( 3 ), 1e6 )
    np.testing.assert_allclose( snCurveFitter.getN( 2 ), -1 )
    np.testing.assert_allclose( snCurveFitter.getN( 1 ), -1 )
    np.testing.assert_allclose( snCurveFitter.getN( 0.3 ), -1 )
    np.testing.assert_allclose( snCurveFitter.getN( 0.2 ), -1 )
    np.testing.assert_allclose( snCurveFitter.getN( 0.1 ), -1 )
