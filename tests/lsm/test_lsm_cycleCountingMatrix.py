#!/usr/bin/env python3

from ffpack import lsm
import numpy as np
import pytest
from unittest.mock import patch


###############################################################################
# Test astmSimpleRangeCountingMatrix
###############################################################################
def test_astmSimpleRangeCountingMatrix_emptyInputCase_valueError():
    # Test edge cases for empty list
    data = [ ]
    with pytest.raises( ValueError ):
        _ = lsm.astmSimpleRangeCountingMatrix( data )


def test_astmSimpleRangeCountingMatrix_singleInputCase_valueError():
    # Test edge cases for 1 element list
    data = [ 1.0 ]
    with pytest.raises( ValueError ):
        _ = lsm.astmSimpleRangeCountingMatrix( data )


def test_astmSimpleRangeCountingMatrix_twoDimInputCase_valueError():
    # Test edge cases for 2D list
    data = [ [ 1.0 ], [ 2.0 ] ]
    with pytest.raises( ValueError ):
        _ = lsm.astmSimpleRangeCountingMatrix( data )


@patch( "ffpack.utils.generalUtils.sequenceDigitization" )
def test_astmSimpleRangeCountingMatrix_astmStandardPoints_matrix( mock_get ):
    data = [ -2.0, 1.0, -3.0, 5.0, -1.0, 3.0, -4.0, 4.0, -2.0 ]
    mock_get.return_value = [ -2.0, 1.0, -3.0, 5.0, -1.0, 3.0, -4.0, 4.0, -2.0 ]
    calMatrix, calKeys = lsm.astmSimpleRangeCountingMatrix( data, resolution=1.0 )
    expectedMatrix = [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0 ],
                       [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5 ],
                       [ 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0 ],
                       [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0 ],
                       [ 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                       [ 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                       [ 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                       [ 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0 ] ]
    expectedKeys = [ -4.0, -3.0, -2.0, -1.0, 1.0, 3.0, 4.0, 5.0 ]
    np.testing.assert_allclose( calMatrix, expectedMatrix )
    np.testing.assert_allclose( calKeys, expectedKeys )

    data = [ -2.0, 1.0, -3.0, 5.0, -1.0, 3.0, -4.0, 4.0, -2.0 ]
    mock_get.return_value = [ -2.0, 1.0, -3.0, 5.0, -1.0, 3.0, -4.0, 4.0, -2.0 ]
    calMatrix, calKeys = lsm.astmSimpleRangeCountingMatrix( data, resolution=0.5 )
    expectedMatrix = [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0 ],
                       [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5 ],
                       [ 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0 ],
                       [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0 ],
                       [ 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                       [ 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                       [ 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                       [ 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0 ] ]
    expectedKeys = [ -4.0, -3.0, -2.0, -1.0, 1.0, 3.0, 4.0, 5.0 ]
    np.testing.assert_allclose( calMatrix, expectedMatrix )
    np.testing.assert_allclose( calKeys, expectedKeys )


@patch( "ffpack.utils.generalUtils.sequenceDigitization" )
def test_astmSimpleRangeCountingMatrix_astmBiasedPoints_matrix( mock_get ):
    data = [ -2.2, 0.8, -3.2, 4.9, -1.1, 3.2, -3.8, 4.1, -2.2 ]
    mock_get.return_value = [ -2.0, 1.0, -3.0, 5.0, -1.0, 3.0, -4.0, 4.0, -2.0 ]
    calMatrix, calKeys = lsm.astmSimpleRangeCountingMatrix( data, resolution=1.0 )
    expectedMatrix = [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0 ],
                       [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5 ],
                       [ 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0 ],
                       [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0 ],
                       [ 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                       [ 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                       [ 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                       [ 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0 ] ]
    expectedKeys = [ -4.0, -3.0, -2.0, -1.0, 1.0, 3.0, 4.0, 5.0 ]
    np.testing.assert_allclose( calMatrix, expectedMatrix )
    np.testing.assert_allclose( calKeys, expectedKeys )

    data = [ -2.2, 0.8, -3.2, 4.9, -1.1, 3.2, -3.8, 4.1, -2.2 ]
    mock_get.return_value = [ -2.0, 1.0, -3.0, 5.0, -1.0, 3.0, -4.0, 4.0, -2.0 ]
    calMatrix, calKeys = lsm.astmSimpleRangeCountingMatrix( data, resolution=0.5 )
    expectedMatrix = [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0 ],
                       [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5 ],
                       [ 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0 ],
                       [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0 ],
                       [ 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                       [ 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                       [ 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                       [ 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0 ] ]
    expectedKeys = [ -4.0, -3.0, -2.0, -1.0, 1.0, 3.0, 4.0, 5.0 ]
    np.testing.assert_allclose( calMatrix, expectedMatrix )
    np.testing.assert_allclose( calKeys, expectedKeys )


@patch( "ffpack.utils.generalUtils.sequenceDigitization" )
def test_astmSimpleRangeCountingMatrix_trivialCases_matrix( mock_get ):
    # Trivial case 1
    data = [ 1.0, 3.0, 2.0 ]
    mock_get.return_value = [ 1.0, 3.0, 2.0 ]
    calMatrix, calKeys = lsm.astmSimpleRangeCountingMatrix( data, resolution=1.0 )
    expectedMatrix = [ [ 0.0, 0.0, 0.5 ],
                       [ 0.0, 0.0, 0.0 ],
                       [ 0.0, 0.5, 0.0 ] ]
    expectedKeys = [ 1.0, 2.0, 3.0 ]
    np.testing.assert_allclose( calMatrix, expectedMatrix )
    np.testing.assert_allclose( calKeys, expectedKeys )

    # Trivial case 2
    data = [ -1.0, -3.0, -2.0 ]
    mock_get.return_value = [ -1.0, -3.0, -2.0 ]
    calMatrix, calKeys = lsm.astmSimpleRangeCountingMatrix( data, resolution=1.0 )
    expectedMatrix = [ [ 0.0, 0.5, 0.0 ],
                       [ 0.0, 0.0, 0.0 ],
                       [ 0.5, 0.0, 0.0 ] ]
    expectedKeys = [ -3.0, -2.0, -1.0 ]
    np.testing.assert_allclose( calMatrix, expectedMatrix )
    np.testing.assert_allclose( calKeys, expectedKeys )

    # Trivial case 3
    data = [ 1.3, 2.6, 1.8 ]
    mock_get.return_value = [ 1.5, 2.5, 2.0 ]
    calMatrix, calKeys = lsm.astmSimpleRangeCountingMatrix( data, resolution=0.5)
    expectedMatrix = [ [ 0.0, 0.0, 0.5 ],
                       [ 0.0, 0.0, 0.0 ],
                       [ 0.0, 0.5, 0.0 ] ]
    expectedKeys = [ 1.5, 2.0, 2.5 ]
    np.testing.assert_allclose( calMatrix, expectedMatrix )
    np.testing.assert_allclose( calKeys, expectedKeys )

    # Trivial case 4
    data = [ -1.3, -2.6, -1.8 ]
    mock_get.return_value = [ -1.5, -2.5, -2.0 ]
    calMatrix, calKeys = lsm.astmSimpleRangeCountingMatrix( data, resolution=0.5)
    expectedMatrix = [ [ 0.0, 0.5, 0.0 ],
                       [ 0.0, 0.0, 0.0 ],
                       [ 0.5, 0.0, 0.0 ] ]
    expectedKeys = [ -2.5, -2.0, -1.5 ]
    np.testing.assert_allclose( calMatrix, expectedMatrix )
    np.testing.assert_allclose( calKeys, expectedKeys )



###############################################################################
# Test astmRainflowCountingMatrix
###############################################################################
def test_astmRainflowCountingMatrix_emptyInputCase_valueError():
    # Test edge cases for empty list
    data = [ ]
    with pytest.raises( ValueError ):
        _ = lsm.astmRainflowCountingMatrix( data )


def test_astmRainflowCountingMatrix_singleInputCase_valueError():
    # Test edge cases for 1 element list
    data = [ 1.0 ]
    with pytest.raises( ValueError ):
        _ = lsm.astmRainflowCountingMatrix( data )


def test_astmRainflowCountingMatrix_twoDimInputCase_valueError():
    # Test edge cases for 2D list
    data = [ [ 1.0 ], [ 2.0 ] ]
    with pytest.raises( ValueError ):
        _ = lsm.astmRainflowCountingMatrix( data )


@patch( "ffpack.utils.generalUtils.sequenceDigitization" )
def test_astmRainflowCountingMatrix_astmStandardPoints_matrix( mock_get ):
    data = [ -2.0, 1.0, -3.0, 5.0, -1.0, 3.0, -4.0, 4.0, -2.0 ]
    mock_get.return_value = [ -2.0, 1.0, -3.0, 5.0, -1.0, 3.0, -4.0, 4.0, -2.0 ]
    calMatrix, calKeys = lsm.astmRainflowCountingMatrix( data, resolution=1.0 )
    expectedMatrix = [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0 ],
                       [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5 ],
                       [ 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0 ],
                       [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 ],
                       [ 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                       [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                       [ 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                       [ 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] ]
    expectedKeys = [ -4.0, -3.0, -2.0, -1.0, 1.0, 3.0, 4.0, 5.0 ]
    np.testing.assert_allclose( calMatrix, expectedMatrix )
    np.testing.assert_allclose( calKeys, expectedKeys )

    data = [ -2.0, 1.0, -3.0, 5.0, -1.0, 3.0, -4.0, 4.0, -2.0 ]
    mock_get.return_value = [ -2.0, 1.0, -3.0, 5.0, -1.0, 3.0, -4.0, 4.0, -2.0 ]
    calMatrix, calKeys = lsm.astmRainflowCountingMatrix( data, resolution=0.5 )
    expectedMatrix = [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0 ],
                       [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5 ],
                       [ 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0 ],
                       [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 ],
                       [ 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                       [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                       [ 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                       [ 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] ]
    expectedKeys = [ -4.0, -3.0, -2.0, -1.0, 1.0, 3.0, 4.0, 5.0 ]
    np.testing.assert_allclose( calMatrix, expectedMatrix )
    np.testing.assert_allclose( calKeys, expectedKeys )


@patch( "ffpack.utils.generalUtils.sequenceDigitization" )
def test_astmRainflowCountingMatrix_astmBiasedPoints_matrix( mock_get ):
    data = [ -2.2, 0.8, -3.2, 4.9, -1.1, 3.2, -3.8, 4.1, -2.2 ]
    mock_get.return_value = [ -2.0, 1.0, -3.0, 5.0, -1.0, 3.0, -4.0, 4.0, -2.0 ]
    calMatrix, calKeys = lsm.astmRainflowCountingMatrix( data, resolution=1.0 )
    expectedMatrix = [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0 ],
                       [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5 ],
                       [ 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0 ],
                       [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 ],
                       [ 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                       [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                       [ 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                       [ 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] ]
    expectedKeys = [ -4.0, -3.0, -2.0, -1.0, 1.0, 3.0, 4.0, 5.0 ]
    np.testing.assert_allclose( calMatrix, expectedMatrix )
    np.testing.assert_allclose( calKeys, expectedKeys )

    data = [ -2.2, 0.8, -3.2, 4.9, -1.1, 3.2, -3.8, 4.1, -2.2 ]
    mock_get.return_value = [ -2.0, 1.0, -3.0, 5.0, -1.0, 3.0, -4.0, 4.0, -2.0 ]
    calMatrix, calKeys = lsm.astmRainflowCountingMatrix( data, resolution=0.5 )
    expectedMatrix = [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0 ],
                       [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5 ],
                       [ 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0 ],
                       [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 ],
                       [ 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                       [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                       [ 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                       [ 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] ]
    expectedKeys = [ -4.0, -3.0, -2.0, -1.0, 1.0, 3.0, 4.0, 5.0 ]
    np.testing.assert_allclose( calMatrix, expectedMatrix )
    np.testing.assert_allclose( calKeys, expectedKeys )

@patch( "ffpack.utils.generalUtils.sequenceDigitization" )
def test_astmRainflowCountingMatrix_trivialCases_matrix( mock_get ):
    # Trivial case 1
    data = [ 1.0, 3.0, 2.0 ]
    mock_get.return_value = [ 1.0, 3.0, 2.0 ]
    calMatrix, calKeys = lsm.astmRainflowCountingMatrix( data, resolution=1.0 )
    expectedMatrix = [ [ 0.0, 0.0, 0.5 ],
                       [ 0.0, 0.0, 0.0 ],
                       [ 0.0, 0.5, 0.0 ] ]
    expectedKeys = [ 1.0, 2.0, 3.0 ]
    np.testing.assert_allclose( calMatrix, expectedMatrix )
    np.testing.assert_allclose( calKeys, expectedKeys )

    # Trivial case 2
    data = [ -1.0, -3.0, -2.0 ]
    mock_get.return_value = [ -1.0, -3.0, -2.0 ]
    calMatrix, calKeys = lsm.astmRainflowCountingMatrix( data, resolution=1.0 )
    expectedMatrix = [ [ 0.0, 0.5, 0.0 ],
                       [ 0.0, 0.0, 0.0 ],
                       [ 0.5, 0.0, 0.0 ] ]
    expectedKeys = [ -3.0, -2.0, -1.0 ]
    np.testing.assert_allclose( calMatrix, expectedMatrix )
    np.testing.assert_allclose( calKeys, expectedKeys )

    # Trivial case 3
    data = [ 1.3, 2.6, 1.8 ]
    mock_get.return_value = [ 1.5, 2.5, 2.0 ]
    calMatrix, calKeys = lsm.astmRainflowCountingMatrix( data, resolution=0.5)
    expectedMatrix = [ [ 0.0, 0.0, 0.5 ],
                       [ 0.0, 0.0, 0.0 ],
                       [ 0.0, 0.5, 0.0 ] ]
    expectedKeys = [ 1.5, 2.0, 2.5 ]
    np.testing.assert_allclose( calMatrix, expectedMatrix )
    np.testing.assert_allclose( calKeys, expectedKeys )

    # Trivial case 4
    data = [ -1.3, -2.6, -1.8 ]
    mock_get.return_value = [ -1.5, -2.5, -2.0 ]
    calMatrix, calKeys = lsm.astmRainflowCountingMatrix( data, resolution=0.5)
    expectedMatrix = [ [ 0.0, 0.5, 0.0 ],
                       [ 0.0, 0.0, 0.0 ],
                       [ 0.5, 0.0, 0.0 ] ]
    expectedKeys = [ -2.5, -2.0, -1.5 ]
    np.testing.assert_allclose( calMatrix, expectedMatrix )
    np.testing.assert_allclose( calKeys, expectedKeys )



###############################################################################
# Test rychlikRainflowCountingmatrix
###############################################################################
def test_rychlikRainflowCountingMatrix_emptyInputCase_valueError():
    # Test edge cases for empty list
    data = [ ]
    with pytest.raises( ValueError ):
        _ = lsm.rychlikRainflowCountingMatrix( data )


def test_rychlikRainflowCountingMatrix_singleInputCase_valueError():
    # Test edge cases for 1 element list
    data = [ 1.0 ]
    with pytest.raises( ValueError ):
        _ = lsm.rychlikRainflowCountingMatrix( data )


def test_rychlikRainflowCountingMatrix_twoDimInputCase_valueError():
    # Test edge cases for 2D list
    data = [ [ 1.0 ], [ 2.0 ] ]
    with pytest.raises( ValueError ):
        _ = lsm.rychlikRainflowCountingMatrix( data )


@patch( "ffpack.utils.generalUtils.sequenceDigitization" )
def test_rychlikRainflowCountingmatrix_astmStandardPoints_matrix( mock_get ):
    data = [ -2.0, 1.0, -3.0, 5.0, -1.0, 3.0, -4.0, 4.0, -2.0 ]
    mock_get.return_value = [ -2.0, 1.0, -3.0, 5.0, -1.0, 3.0, -4.0, 4.0, -2.0 ]
    calMatrix, calKeys = lsm.rychlikRainflowCountingMatrix( data, resolution=1.0 )
    expectedMatrix = [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 ],
                       [ 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0 ],
                       [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 ],
                       [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                       [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                       [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                       [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] ]
    expectedKeys = [ -3.0, -2.0, -1.0, 1.0, 3.0, 4.0, 5.0 ]
    np.testing.assert_allclose( calMatrix, expectedMatrix )
    np.testing.assert_allclose( calKeys, expectedKeys )

    data = [ -2.0, 1.0, -3.0, 5.0, -1.0, 3.0, -4.0, 4.0, -2.0 ]
    mock_get.return_value = [ -2.0, 1.0, -3.0, 5.0, -1.0, 3.0, -4.0, 4.0, -2.0 ]
    calMatrix, calKeys = lsm.rychlikRainflowCountingMatrix( data, resolution=0.5 )
    expectedMatrix = [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 ],
                       [ 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0 ],
                       [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 ],
                       [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                       [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                       [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                       [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] ]
    expectedKeys = [ -3.0, -2.0, -1.0, 1.0, 3.0, 4.0, 5.0 ]
    np.testing.assert_allclose( calMatrix, expectedMatrix )
    np.testing.assert_allclose( calKeys, expectedKeys )


@patch( "ffpack.utils.generalUtils.sequenceDigitization" )
def test_rychlikRainflowCountingmatrix_astmBiasedPoints_matrix( mock_get ):
    data = [ -2.2, 0.8, -3.2, 4.9, -1.1, 3.2, -3.8, 4.1, -2.2 ]
    mock_get.return_value = [ -2.0, 1.0, -3.0, 5.0, -1.0, 3.0, -4.0, 4.0, -2.0 ]
    calMatrix, calKeys = lsm.rychlikRainflowCountingMatrix( data, resolution=1.0 )
    expectedMatrix = [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 ],
                       [ 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0 ],
                       [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 ],
                       [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                       [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                       [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                       [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] ]
    expectedKeys = [ -3.0, -2.0, -1.0, 1.0, 3.0, 4.0, 5.0 ]
    np.testing.assert_allclose( calMatrix, expectedMatrix )
    np.testing.assert_allclose( calKeys, expectedKeys )

    data = [ -2.2, 0.8, -3.2, 4.9, -1.1, 3.2, -3.8, 4.1, -2.2 ]
    mock_get.return_value = [ -2.0, 1.0, -3.0, 5.0, -1.0, 3.0, -4.0, 4.0, -2.0 ]
    calMatrix, calKeys = lsm.rychlikRainflowCountingMatrix( data, resolution=0.5 )
    expectedMatrix = [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 ],
                       [ 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0 ],
                       [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 ],
                       [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                       [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                       [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                       [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] ]
    expectedKeys = [ -3.0, -2.0, -1.0, 1.0, 3.0, 4.0, 5.0 ]
    np.testing.assert_allclose( calMatrix, expectedMatrix )
    np.testing.assert_allclose( calKeys, expectedKeys )

@patch( "ffpack.utils.generalUtils.sequenceDigitization" )
def test_rychlikRainflowCountingMatrix_trivialCases_matrix( mock_get ):
    # Trivial case 1
    data = [ 1.0, 3.0, 2.0 ]
    mock_get.return_value = [ 1.0, 3.0, 2.0 ]
    calMatrix, calKeys = lsm.rychlikRainflowCountingMatrix( data, resolution=1.0 )
    expectedMatrix = [ [ 0.0, 1.0 ],
                       [ 0.0, 0.0 ] ]
    expectedKeys = [ 2.0, 3.0 ]
    np.testing.assert_allclose( calMatrix, expectedMatrix )
    np.testing.assert_allclose( calKeys, expectedKeys )

    # Trivial case 2
    data = [ -1.0, -3.0, -2.0 ]
    mock_get.return_value = [ -1.0, -3.0, -2.0 ]
    calMatrix, calKeys = lsm.rychlikRainflowCountingMatrix( data, resolution=1.0 )
    expectedMatrix = [ [ ] ]
    expectedKeys = [ ]
    np.testing.assert_allclose( calMatrix, expectedMatrix )
    np.testing.assert_allclose( calKeys, expectedKeys )

    # Trivial case 3
    data = [ 1.3, 2.6, 1.8 ]
    mock_get.return_value = [ 1.5, 2.5, 2.0 ]
    calMatrix, calKeys = lsm.rychlikRainflowCountingMatrix( data, resolution=0.5)
    expectedMatrix = [ [ 0.0, 1.0 ],
                       [ 0.0, 0.0 ] ]
    expectedKeys = [ 2.0, 2.5 ]
    np.testing.assert_allclose( calMatrix, expectedMatrix )
    np.testing.assert_allclose( calKeys, expectedKeys )

    # # Trivial case 4
    data = [ -1.3, -2.6, -1.8 ]
    mock_get.return_value = [ -1.5, -2.5, -2.0 ]
    calMatrix, calKeys = lsm.rychlikRainflowCountingMatrix( data, resolution=0.5)
    expectedMatrix = [ [ ] ]
    expectedKeys = [ ]
    np.testing.assert_allclose( calMatrix, expectedMatrix )
    np.testing.assert_allclose( calKeys, expectedKeys )
