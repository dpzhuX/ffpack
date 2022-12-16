#!/usr/bin/env python3

from ffpack import lsg
import numpy as np
import pytest
from unittest.mock import patch


###############################################################################
# Test arNormal
###############################################################################
def test_arNormal_numStepsLessThanOneCase_valueError():
    obs = [ 0, 0  ]
    phis = [ 0.5, 0.3 ]
    with pytest.raises( ValueError ):
        _ = lsg.arNormal( -1, obs, phis, 0, 0.5 )

    with pytest.raises( ValueError ):
        _ = lsg.arNormal( 0, obs, phis, 0, 0.5 )


def test_arNormal_lengthsNotEqualCase_valueError():
    obs = [ 0, 0  ]
    phis = [ 0.3 ]
    with pytest.raises( ValueError ):
        _ = lsg.arNormal( 500, obs, phis, 0, 0.5 )

    obs = [ 0 ]
    phis = [ 0.3, 0.5 ]
    with pytest.raises( ValueError ):
        _ = lsg.arNormal( 500, obs, phis, 0, 0.5 )


def test_arNormal_enptyObsAndPhisCase_valueError():
    obs = [ ]
    phis = [ ]
    with pytest.raises( ValueError ):
        _ = lsg.arNormal( 500, obs, phis, 0, 0.5 )


@patch( "numpy.random.normal" )
def test_arNormal_ar2ThreeStepsCase_threePointsOutput( mock_get ):
    obs = [ 0, 0 ]
    phis = [ 0.5, 0.3 ]
    mock_get.return_value = [ 0.0, 1.0, 2.0 ]
    calRst = lsg.arNormal( 3, obs, phis, 0, 0.5 )
    expectedRst = [ 0, 0, 2.0 ]
    np.testing.assert_allclose( calRst, expectedRst )

    obs = [ 1.0, 1.0 ]
    phis = [ 0.5, 0.3 ]
    mock_get.return_value = [ 2.0, 1.0, 0.0 ]
    calRst = lsg.arNormal( 3, obs, phis, 0, 0.5 )
    expectedRst = [ 1.0, 1.0, 0.8 ]
    np.testing.assert_allclose( calRst, expectedRst )

    obs = [ 1.0, 1.0 ]
    phis = [ 0.5, 0.5 ]
    mock_get.return_value = [ 2.0, 1.0, 1.0 ]
    calRst = lsg.arNormal( 3, obs, phis, 0, 0.5 )
    expectedRst = [ 1.0, 1.0, 2.0 ]
    np.testing.assert_allclose( calRst, expectedRst )

@patch( "numpy.random.normal" )
def test_arNormal_ar2FourStepsCase_threePointsOutput( mock_get ):
    obs = [ 0, 0 ]
    phis = [ 0.5, 0.3 ]
    mock_get.return_value = [ 0.0, 1.0, 2.0, 3.0 ]
    calRst = lsg.arNormal( 4, obs, phis, 0, 0.5 )
    expectedRst = [ 0, 0, 2.0, 4.0 ]
    np.testing.assert_allclose( calRst, expectedRst )

    obs = [ 1.0, 1.0 ]
    phis = [ 0.5, 0.3 ]
    mock_get.return_value = [ 2.0, 1.0, 0.0, 0.5 ]
    calRst = lsg.arNormal( 4, obs, phis, 0, 0.5 )
    expectedRst = [ 1.0, 1.0, 0.8, 1.2 ]
    np.testing.assert_allclose( calRst, expectedRst )

    obs = [ 1.0, 1.0 ]
    phis = [ 0.5, 0.5 ]
    mock_get.return_value = [ 2.0, 1.0, 1.0, -2.0 ]
    calRst = lsg.arNormal( 4, obs, phis, 0, 0.5 )
    expectedRst = [ 1.0, 1.0, 2.0, -0.5 ]
    np.testing.assert_allclose( calRst, expectedRst )

    obs = [ 1.0, 0.5 ]
    phis = [ 1.0, 0.5 ]
    mock_get.return_value = [ 2.0, -1.0, -1.0, 2.0 ]
    calRst = lsg.arNormal( 4, obs, phis, 0, 0.5 )
    expectedRst = [ 1.0, 0.5, 0, 2.25 ]
    np.testing.assert_allclose( calRst, expectedRst )

    obs = [ 1.0, 0.5 ]
    phis = [ 1.0, -0.5 ]
    mock_get.return_value = [ 2.0, 1.0, -1.0, -2.0 ]
    calRst = lsg.arNormal( 4, obs, phis, 0, 0.5 )
    expectedRst = [ 1.0, 0.5, -1.0, -3.25 ]
    np.testing.assert_allclose( calRst, expectedRst )
