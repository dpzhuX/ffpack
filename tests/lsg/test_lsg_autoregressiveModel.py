#!/usr/bin/env python3

from ffpack import lsg
import numpy as np
import pytest
from unittest.mock import patch


###############################################################################
# Test arNormal
###############################################################################
def test_arNormal_numStepsLessThanOneCase_valueError():
    obs = [ 0, 0 ]
    phis = [ 0.5, 0.3 ]
    with pytest.raises( ValueError ):
        _ = lsg.arNormal( -1, obs, phis, 0, 0.5 )

    with pytest.raises( ValueError ):
        _ = lsg.arNormal( 0, obs, phis, 0, 0.5 )


def test_arNormal_lengthsNotEqualCase_valueError():
    obs = [ 0, 0 ]
    phis = [ 0.3 ]
    with pytest.raises( ValueError ):
        _ = lsg.arNormal( 500, obs, phis, 0, 0.5 )

    obs = [ 0 ]
    phis = [ 0.3, 0.5 ]
    with pytest.raises( ValueError ):
        _ = lsg.arNormal( 500, obs, phis, 0, 0.5 )


def test_arNormal_emptyObsAndPhisCase_valueError():
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



###############################################################################
# Test maNormal
###############################################################################
def test_maNormal_numStepsNotInt_valueError():
    thetas = [ 0.5, 0.2 ]
    with pytest.raises( ValueError ):
        _ = lsg.maNormal( 1.2, 0, thetas, 0, 0.5 )


def test_maNormal_numStepsLessThanOneCase_valueError():
    thetas = [ 0.5, 0.2 ]
    with pytest.raises( ValueError ):
        _ = lsg.maNormal( -1, 0, thetas, 0, 0.5 )

    with pytest.raises( ValueError ):
        _ = lsg.maNormal( 0, 0, thetas, 0, 0.5 )


def test_maNormal_thetasEmptyCase_valueError():
    thetas = [  ]
    with pytest.raises( ValueError ):
        _ = lsg.maNormal( 500, 0, thetas, 0, 0.5 )


@patch( "numpy.random.normal" )
def test_maNormal_oneStepCase_outputNotRelatedToThetas( mock_get ):
    mock_get.return_value = [ 1.0 ]

    thetas = [ 0.5 ]
    calRst = lsg.maNormal( 1, 0, thetas, 0, 0.5 )
    expectedRst = [ 1.0 ]
    np.testing.assert_allclose( calRst, expectedRst )

    thetas = [ 0.5, 0.2 ]
    calRst = lsg.maNormal( 1, 0, thetas, 0, 0.5 )
    expectedRst = [ 1.0 ]
    np.testing.assert_allclose( calRst, expectedRst )


@patch( "numpy.random.normal" )
def test_maNormal_twoStepCase_outputRelatedToOneTheta( mock_get ):
    thetas = [ 0.5 ]
    mock_get.return_value = [ 1.0, 2.0 ]
    calRst = lsg.maNormal( 2, 0, thetas, 0, 0.5 )
    expectedRst = [ 1.0, 2.5 ]
    np.testing.assert_allclose( calRst, expectedRst )

    thetas = [ 0.5, 0.2 ]
    mock_get.return_value = [ 1.0, 2.0 ]
    calRst = lsg.maNormal( 2, 0, thetas, 0, 0.5 )
    expectedRst = [ 1.0, 2.5 ]
    np.testing.assert_allclose( calRst, expectedRst )


@patch( "numpy.random.normal" )
def test_maNormal_threeStepCase_outputRelatedToTwoThetas( mock_get ):
    mock_get.return_value = [ 1.0, 2.0, 3.0 ]

    thetas = [ 0.5 ]
    calRst = lsg.maNormal( 3, 0, thetas, 0, 0.5 )
    expectedRst = [ 1.0, 2.5, 4.0 ]
    np.testing.assert_allclose( calRst, expectedRst )

    thetas = [ 0.5, 0.2 ]
    calRst = lsg.maNormal( 3, 0, thetas, 0, 0.5 )
    expectedRst = [ 1.0, 2.5, 4.2 ]
    np.testing.assert_allclose( calRst, expectedRst )

    thetas = [ 0.5, 0.2, 0.1 ]
    calRst = lsg.maNormal( 3, 0, thetas, 0, 0.5 )
    expectedRst = [ 1.0, 2.5, 4.2 ]
    np.testing.assert_allclose( calRst, expectedRst )


@patch( "numpy.random.normal" )
def test_maNormal_fourStepCase_outputRelatedToThreeThetas( mock_get ):
    mock_get.return_value = [ 1.0, 2.0, 3.0, 4.0 ]

    thetas = [ 0.8 ]
    calRst = lsg.maNormal( 4, 0, thetas, 0, 0.5 )
    expectedRst = [ 1.0, 2.8, 4.6, 6.4 ]
    np.testing.assert_allclose( calRst, expectedRst )

    thetas = [ 0.8, 0.5 ]
    calRst = lsg.maNormal( 4, 0, thetas, 0, 0.5 )
    expectedRst = [ 1.0, 2.8, 5.1, 7.4 ]
    np.testing.assert_allclose( calRst, expectedRst )

    thetas = [ 0.8, 0.5, 0.2 ]
    calRst = lsg.maNormal( 4, 0, thetas, 0, 0.5 )
    expectedRst = [ 1.0, 2.8, 5.1, 7.6 ]
    np.testing.assert_allclose( calRst, expectedRst )

    thetas = [ 0.8, 0.5, 0.2, 0.1 ]
    calRst = lsg.maNormal( 4, 0, thetas, 0, 0.5 )
    expectedRst = [ 1.0, 2.8, 5.1, 7.6 ]
    np.testing.assert_allclose( calRst, expectedRst )



###############################################################################
# Test armaNormal
###############################################################################
def test_armaNormal_numStepsNotInt_valueError():
    obs = [ 0, 0 ]
    phis = [ 0.5, 0.3 ]
    thetas = [ 0.5, 0.2 ]
    with pytest.raises( ValueError ):
        _ = lsg.armaNormal( 1.2, obs, phis, thetas, 0, 0.5 )


def test_armaNormal_numStepsLessThanOneCase_valueError():
    obs = [ 0, 0 ]
    phis = [ 0.5, 0.3 ]
    thetas = [ 0.5, 0.2 ]
    with pytest.raises( ValueError ):
        _ = lsg.armaNormal( -1, obs, phis, thetas, 0, 0.5 )

    with pytest.raises( ValueError ):
        _ = lsg.armaNormal( 0, obs, phis, thetas, 0, 0.5 )


def test_armaNormal_PhisEmptyCase_valueError():
    obs = [ ]
    phis = [ ]
    thetas = [ 0.5, 0.2 ]
    with pytest.raises( ValueError ):
        _ = lsg.armaNormal( 500, obs, phis, thetas, 0, 0.5 )


def test_armaNormal_thetasEmptyCase_valueError():
    obs = [ 0, 0 ]
    phis = [ 0.5, 0.3 ]
    thetas = [ ]
    with pytest.raises( ValueError ):
        _ = lsg.armaNormal( 0, obs, phis, thetas, 0, 0.5 )


@patch( "numpy.random.normal" )
def test_armaNormal_oneStepCase_outputNotRelatedToPhisAndThetas( mock_get ):
    mock_get.return_value = [ 1.0 ]
    
    phis = [ 0.5, 0.3 ]
    thetas = [ 0.8, 0.5 ]

    # with obs
    obs = [ 2.0, 3.0 ]
    calRst = lsg.armaNormal( 1, obs, phis, thetas, 0, 0.5 )
    expectedRst = [ 2.0 ]
    np.testing.assert_allclose( calRst, expectedRst )

    # without obs
    obs = [ ]
    calRst = lsg.armaNormal( 1, obs, phis, thetas, 0, 0.5 )
    expectedRst = [ 1.0 ]
    np.testing.assert_allclose( calRst, expectedRst )


@patch( "numpy.random.normal" )
def test_armaNormal_twoStepCase_outputDepends( mock_get ):
    mock_get.return_value = [ 1.0, 3.0 ]
    
    phis = [ 0.5, 0.3 ]
    thetas = [ 0.8, 0.5 ]

    # with two obs points
    obs = [ 2.0, 3.0 ]
    calRst = lsg.armaNormal( 2, obs, phis, thetas, 0, 0.5 )
    expectedRst = [ 2.0, 3.0 ]
    np.testing.assert_allclose( calRst, expectedRst )

    # with one obs point
    obs = [ 2.0 ]
    calRst = lsg.armaNormal( 2, obs, phis, thetas, 0, 0.5 )
    expectedRst = [ 2.0, 4.8 ]
    np.testing.assert_allclose( calRst, expectedRst )

    # without obs
    obs = [ ]
    calRst = lsg.armaNormal( 2, obs, phis, thetas, 0, 0.5 )
    expectedRst = [ 1.0, 4.3 ]
    np.testing.assert_allclose( calRst, expectedRst )


@patch( "numpy.random.normal" )
def test_armaNormal_threeStepCase_outputDepends( mock_get ):
    mock_get.return_value = [ 1.0, 3.0, 4.0 ]
    
    # case 1: enough phis and thetas
    phis = [ 0.5, 0.3, 0.2 ]
    thetas = [ 0.8, 0.5, 0.4 ]
    # with three obs points
    obs = [ 2.0, 3.0, 4.0 ]
    calRst = lsg.armaNormal( 3, obs, phis, thetas, 0, 0.5 )
    expectedRst = [ 2.0, 3.0, 4.0 ]
    np.testing.assert_allclose( calRst, expectedRst )
    # with two obs points
    obs = [ 2.0, 3.0 ]
    calRst = lsg.armaNormal( 3, obs, phis, thetas, 0, 0.5 )
    expectedRst = [ 2.0, 3.0, 9.0 ]
    np.testing.assert_allclose( calRst, expectedRst )
    # with one obs point
    obs = [ 2.0 ]
    calRst = lsg.armaNormal( 3, obs, phis, thetas, 0, 0.5 )
    expectedRst = [ 2.0, 4.8, 9.9 ]
    np.testing.assert_allclose( calRst, expectedRst )
    # without obs
    obs = [ ]
    calRst = lsg.armaNormal( 3, obs, phis, thetas, 0, 0.5 )
    expectedRst = [ 1.0, 4.3, 9.35 ]
    np.testing.assert_allclose( calRst, expectedRst )

    # case 2: one phis and one thetas
    phis = [ 0.5 ]
    thetas = [ 0.8 ]
    # with two obs points
    obs = [ 2.0, 3.0 ]
    calRst = lsg.armaNormal( 3, obs, phis, thetas, 0, 0.5 )
    expectedRst = [ 2.0, 3.0, 7.9 ]
    np.testing.assert_allclose( calRst, expectedRst )
    # with one obs point
    obs = [ 2.0 ]
    calRst = lsg.armaNormal( 3, obs, phis, thetas, 0, 0.5 )
    expectedRst = [ 2.0, 4.8, 8.8 ]
    np.testing.assert_allclose( calRst, expectedRst )
    # without obs
    obs = [ ]
    calRst = lsg.armaNormal( 3, obs, phis, thetas, 0, 0.5 )
    expectedRst = [ 1.0, 4.3, 8.55 ]
    np.testing.assert_allclose( calRst, expectedRst )

    # case 3: one phis and two thetas
    phis = [ 0.5 ]
    thetas = [ 0.8, 0.5 ]
    # with two obs points
    obs = [ 2.0, 3.0 ]
    calRst = lsg.armaNormal( 3, obs, phis, thetas, 0, 0.5 )
    expectedRst = [ 2.0, 3.0, 8.4 ]
    np.testing.assert_allclose( calRst, expectedRst )
    # with one obs point
    obs = [ 2.0 ]
    calRst = lsg.armaNormal( 3, obs, phis, thetas, 0, 0.5 )
    expectedRst = [ 2.0, 4.8, 9.3 ]
    np.testing.assert_allclose( calRst, expectedRst )
    # without obs
    obs = [ ]
    calRst = lsg.armaNormal( 3, obs, phis, thetas, 0, 0.5 )
    expectedRst = [ 1.0, 4.3, 9.05 ]
    np.testing.assert_allclose( calRst, expectedRst )

    # case 3: two phis and one thetas
    phis = [ 0.5, 0.3 ]
    thetas = [ 0.8 ]
    # with two obs points
    obs = [ 2.0, 3.0 ]
    calRst = lsg.armaNormal( 3, obs, phis, thetas, 0, 0.5 )
    expectedRst = [ 2.0, 3.0, 8.5 ]
    np.testing.assert_allclose( calRst, expectedRst )
    # with one obs point
    obs = [ 2.0 ]
    calRst = lsg.armaNormal( 3, obs, phis, thetas, 0, 0.5 )
    expectedRst = [ 2.0, 4.8, 9.4 ]
    np.testing.assert_allclose( calRst, expectedRst )
    # without obs
    obs = [ ]
    calRst = lsg.armaNormal( 3, obs, phis, thetas, 0, 0.5 )
    expectedRst = [ 1.0, 4.3, 8.85 ]
    np.testing.assert_allclose( calRst, expectedRst )
